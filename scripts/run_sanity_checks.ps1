$ErrorActionPreference = "Stop"

function Invoke-Step {
    param(
        [string]$Title,
        [string[]]$CommandArgs
    )

    Write-Host ""
    Write-Host "== $Title ==" -ForegroundColor Cyan
    & python @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Sanity check failed at step: $Title"
    }
}

Invoke-Step -Title "Pytest" -CommandArgs @("-m", "pytest", "tests", "-vv")
Invoke-Step -Title "Verify Python Environment" -CommandArgs @("-m", "scripts.verify_python_env")
Invoke-Step -Title "Runtime Environment Report" -CommandArgs @("-m", "scripts.check_runtime_env")
Invoke-Step -Title "Depth Projection Demo" -CommandArgs @("-m", "scripts.run_depth_projection_demo")
Invoke-Step -Title "Validate Replay Scene" -CommandArgs @("-m", "scripts.validate_scene_dir", "--scene_dir", "data/scenes/scene_01")
Invoke-Step -Title "Scene State Demo" -CommandArgs @("-m", "scripts.run_scene_state_demo", "--scene_dir", "data/scenes/scene_01", "--backend", "demo")
Invoke-Step -Title "Pick Plan Demo" -CommandArgs @("-m", "scripts.run_pick_plan_demo", "--scene_dir", "data/scenes/scene_01", "--backend", "demo")
Invoke-Step -Title "FSM Dry Run" -CommandArgs @("-m", "scripts.run_fsm_once", "--use_fake_robot", "--scene_dir", "data/scenes/scene_01", "--backend", "demo")

Write-Host ""
Write-Host "All Windows-Dev sanity checks passed." -ForegroundColor Green
