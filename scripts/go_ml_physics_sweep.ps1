$ErrorActionPreference = "Stop"

Write-Host "=========================================================="
Write-Host " Starting ML/Physics Architecture Sweep"
Write-Host "=========================================================="

Write-Host "`n[1/5] Level 1.1: Global Physics Parameter tuning..."
python scripts/tune_differentiable_wall_model.py --epochs 25 --lr 0.05 --flow pred --model-type global --save "outputs/sweep_level1_1.json"

Write-Host "`n[2/5] Level 1.2: Local GNN Parameter prediction..."
python scripts/tune_differentiable_wall_model.py --epochs 25 --lr 0.005 --flow pred --model-type local --save "outputs/sweep_level1_2.json"

Write-Host "`n[3/5] Level 2: GNO for Growth Dynamics..."
python scripts/tune_differentiable_wall_model.py --epochs 25 --lr 0.005 --flow pred --model-type global --use-growth-gno --save "outputs/sweep_level2_gno.json"

Write-Host "`n[4/5] Level 3: Deep Chemical State Estimator..."
python scripts/tune_differentiable_wall_model.py --epochs 25 --lr 0.005 --flow pred --model-type global --use-chem-estimator --save "outputs/sweep_level3_chem.json"

Write-Host "`n[5/5] The Kitchen Sink: All ML modules active..."
python scripts/tune_differentiable_wall_model.py --epochs 25 --lr 0.005 --flow pred --model-type local --use-growth-gno --use-chem-estimator --save "outputs/sweep_kitchen_sink.json"

Write-Host "`n=========================================================="
Write-Host " Sweep Complete!"
Write-Host "=========================================================="
