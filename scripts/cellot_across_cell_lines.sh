#! /bin/bash

# Convenience launcher that prints all CellOT commands.
# Recommended order:
#   1) Run commands printed by cellot_across_cell_lines_train_scgen.sh.
#   2) After all AE jobs finish, run commands printed by cellot_across_cell_lines_train_cellot.sh.
#   3) After all CellOT jobs finish, run commands printed by cellot_across_cell_lines_predict.sh.

bash cellot_across_cell_lines_train_scgen.sh
bash cellot_across_cell_lines_train_cellot.sh
bash cellot_across_cell_lines_predict.sh
