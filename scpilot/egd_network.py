import torch
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl
from .encoder import Encoder
from .generator import Generator
from .discriminator import Discriminator
class EGD_network(torch.nn.Module):
    def __init__(
        self,
        n_input: int = 9999,
        n_latent: int = 100,
        n_layers: int = 2,
        n_hidden: int = 800,
        dropout_rate: float = 0.2,
        latent_distribution: str = 'normal',
        lambd1: float = 0.2,
        lambd2: float = 1.0,
        lambd3: float = 1e-3,
        lambd4: float = 0.05,
        use_discriminator: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_latent = n_latent
        self.latent_distribution = 'normal'
        self.lambd1 = lambd1
        self.lambd2 = lambd2
        self.lambd3 = lambd3
        self.lambd4 = lambd4
        self.use_discriminator = use_discriminator
        self.eps = eps
        self.z_encoder = Encoder(
            n_input,
            n_latent,
            n_layers = n_layers,
            n_hidden = n_hidden,
            dropout_rate = dropout_rate,
            distribution = latent_distribution,
            activation_fn = torch.nn.LeakyReLU,
        )
        n_input_generator = n_latent
        self.generator = Generator(
            n_input_generator,
            n_input,
            n_layers = n_layers,
            n_hidden = n_hidden,
            dropout_rate = dropout_rate,
            activation_fn = torch.nn.LeakyReLU,
        )
        if self.use_discriminator:
            self.discriminator = Discriminator(
                n_input,
                1,
                n_layers=n_layers,
                n_hidden=n_hidden,
                dropout_rate=dropout_rate,
                activation_fn=torch.nn.LeakyReLU,
            )
        else:
            self.discriminator = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.to(self.device)
        q_m, q_v, z = self.z_encoder(x)
        return q_m, q_v, z
    def generate(self, z: torch.Tensor) -> torch.Tensor:
        z = z.to(self.device)
        xHat = self.generator(z)
        return xHat
    def discriminate(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.discriminator is None:
            raise RuntimeError(
                "The discriminator is disabled for this model."
            )

        x = x.to(self.device)
        f_D, proba = self.discriminator(x)
        return f_D, proba
    def get_reconstruction_loss(self, x: torch.Tensor, xHat: torch.Tensor) -> torch.Tensor:
        loss = ((x - xHat) ** 2).sum(dim = 1)
        return loss
    def loss(
        self,
        x: torch.Tensor,
        q_m: torch.Tensor,
        q_v: torch.Tensor,
        xHat: torch.Tensor,
        f_D_x: torch.Tensor = None,
        f_D_xHat: torch.Tensor = None,
        f_D_xHat_p: torch.Tensor = None,
        proba_x: torch.Tensor = None,
        proba_xHat: torch.Tensor = None,
        proba_xHat_p: torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:

        # VAE KL divergence.
        kld = kl(
            Normal(q_m, torch.sqrt(q_v)),
            Normal(0, 1),
        ).sum(dim=1)

        # Gene-space reconstruction loss, shared by both variants.
        gene_reconstruction_loss = self.get_reconstruction_loss(
            xHat,
            x,
        )

        if self.use_discriminator:
            if any(
                value is None
                for value in [
                    f_D_x,
                    f_D_xHat,
                    f_D_xHat_p,
                    proba_x,
                    proba_xHat,
                    proba_xHat_p,
                ]
            ):
                raise ValueError(
                    "Discriminator outputs are required when "
                    "use_discriminator=True."
                )

            # Reconstruction in both gene space and discriminator feature space.
            rl = self.lambd2 * (
                gene_reconstruction_loss
                + self.get_reconstruction_loss(
                    f_D_xHat,
                    f_D_x,
                )
            )

            # Feature matching between prior-generated and real cells.
            rl = rl + self.lambd3 * self.get_reconstruction_loss(
                f_D_xHat_p.mean(dim=0, keepdim=True),
                f_D_x.mean(dim=0, keepdim=True),
            )

            # Discriminator classification loss.
            dl = -(
                torch.log10(proba_x + self.eps)
                + torch.log10(1 - proba_xHat + self.eps)
                + torch.log10(1 - proba_xHat_p + self.eps)
            )
            dl = (self.lambd4 * dl).mean()

        else:
            # Strict w/o-discriminator variant:
            # no feature matching and no adversarial classification.
            rl = self.lambd2 * gene_reconstruction_loss

            # Keep a scalar zero for unified logging.
            dl = torch.zeros(
                (),
                device=x.device,
                dtype=x.dtype,
            )

        VAE_loss = (
            0.5 * self.lambd1 * kld
            + 0.5 * rl
        ).mean()

        return {
            'VAE_loss': VAE_loss,
            'rl': rl,
            'kld': kld,
            'dl': dl,
        }
    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        x = x.to(self.device)

        q_m, q_v, z = self.encode(x)
        xHat = self.generate(z)

        inference_outputs = {
            'q_m': q_m,
            'q_v': q_v,
            'z': z,
        }

        generative_outputs = {
            'xHat': xHat,
        }

        if self.use_discriminator:
            z_p = Normal(0, 1).sample(z.size()).to(self.device)
            xHat_p = self.generate(z_p)

            f_D_x, proba_x = self.discriminate(x)
            f_D_xHat, proba_xHat = self.discriminate(xHat)
            f_D_xHat_p, proba_xHat_p = self.discriminate(xHat_p)

            generative_outputs['xHat_p'] = xHat_p

            discriminator_outputs = {
                'f_D_x': f_D_x,
                'f_D_xHat': f_D_xHat,
                'f_D_xHat_p': f_D_xHat_p,
                'proba_x': proba_x,
                'proba_xHat': proba_xHat,
                'proba_xHat_p': proba_xHat_p,
            }

            losses = self.loss(
                x=x,
                q_m=q_m,
                q_v=q_v,
                xHat=xHat,
                f_D_x=f_D_x,
                f_D_xHat=f_D_xHat,
                f_D_xHat_p=f_D_xHat_p,
                proba_x=proba_x,
                proba_xHat=proba_xHat,
                proba_xHat_p=proba_xHat_p,
            )

        else:
            discriminator_outputs = {}

            losses = self.loss(
                x=x,
                q_m=q_m,
                q_v=q_v,
                xHat=xHat,
            )

        return (
            inference_outputs,
            generative_outputs,
            discriminator_outputs,
            losses,
        )