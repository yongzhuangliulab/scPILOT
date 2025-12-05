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
        eps = 1e-6,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_latent = n_latent
        self.latent_distribution = 'normal'
        self.lambd1 = lambd1
        self.lambd2 = lambd2
        self.lambd3 = lambd3
        self.lambd4 = lambd4
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
        self.discriminator = Discriminator(
            n_input,
            1,
            n_layers = n_layers,
            n_hidden = n_hidden,
            dropout_rate = dropout_rate,
            activation_fn = torch.nn.LeakyReLU,
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.to(self.device)
        q_m, q_v, z = self.z_encoder(x)
        return q_m, q_v, z
    def generate(self, z: torch.Tensor) -> torch.Tensor:
        z = z.to(self.device)
        xHat = self.generator(z)
        return xHat
    def discriminate(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
        f_D_x: torch.Tensor,
        f_D_xHat: torch.Tensor,
        f_D_xHat_p: torch.Tensor,
        proba_x: torch.Tensor,
        proba_xHat: torch.Tensor,
        proba_xHat_p: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        kld = kl(
            Normal(q_m, torch.sqrt(q_v)),
            Normal(0, 1),
        ).sum(dim = 1)
        rl = self.lambd2 * (
                self.get_reconstruction_loss(xHat, x) + 
                self.get_reconstruction_loss(f_D_xHat, f_D_x)
            ) + self.lambd3 * self.get_reconstruction_loss(
                f_D_xHat_p.mean(dim = 0, keepdim = True), f_D_x.mean(dim = 0, keepdim = True)
            )
        dl = -(torch.log10(proba_x + self.eps) + torch.log10(1 - proba_xHat + self.eps) + torch.log10(1 - proba_xHat_p + self.eps))
        dl = (self.lambd4 * dl).mean()
        VAE_loss = (0.5 * self.lambd1 * kld + 0.5 * rl).mean()
        return {
            'VAE_loss': VAE_loss,
            'rl': rl,
            'kld': kld,
            'dl': dl,
        }
    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        x = x.to(self.device)
        q_m, q_v, z = self.encode(x)
        z_p = Normal(0, 1).sample(z.size()).to(self.device)
        xHat = self.generate(z)
        xHat_p = self.generate(z_p)
        f_D_x, proba_x = self.discriminate(x)
        f_D_xHat, proba_xHat = self.discriminate(xHat)
        f_D_xHat_p, proba_xHat_p = self.discriminate(xHat_p)
        losses = self.loss(x, q_m, q_v, xHat, f_D_x, f_D_xHat, f_D_xHat_p, proba_x, proba_xHat, proba_xHat_p)
        return {
            'q_m': q_m, 'q_v': q_v, 'z': z
        }, {
            'xHat': xHat, 'xHat_p': xHat_p
        }, {
            'f_D_x': f_D_x, 'f_D_xHat': f_D_xHat, 'f_D_xHat_p': f_D_xHat_p, 'proba_x': proba_x, 'proba_xHat': proba_xHat, 'proba_xHat_p': proba_xHat_p
        }, losses