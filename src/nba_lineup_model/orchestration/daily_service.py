"""Start a daily worker or manual run from sealed code and its bound instance."""
import argparse
import json
import os
import sys
from pathlib import Path

from nba_lineup_model.orchestration.daily_instance import DailyInstance, execute_manual
from nba_lineup_model.orchestration.daily_work import read_object
from nba_lineup_model.season.sealed_code_package import verify_sealed_code_package


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--package", type=Path, required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--home", type=Path, required=True)
    p.add_argument("--environment", choices=("dev", "prod"))
    p.add_argument("--component", choices=("code", "daemon", "web", "manual"), required=True)
    p.add_argument("--grpc-port", type=int, required=True)
    p.add_argument("--web-port", type=int, required=True)
    p.add_argument("--job", choices=("prepare_daily_partition", "prepare_forecast_date", "advance_daily_environment"))
    p.add_argument("--request", type=Path)
    args = p.parse_args()
    manifest = verify_sealed_code_package(args.package)
    code = args.package.resolve() / "code"
    if manifest["identity"]["package_sha256"] != args.sha256 or not Path(__file__).resolve().is_relative_to(code):
        raise ValueError("Daily worker is not executing the selected sealed code")
    metadata = read_object(args.home / "environment.json")
    selected = args.environment
    if metadata.get("schema_version") == 2:
        if args.component in {"code", "manual"} and not selected:
            p.error("A shared code location/manual run requires --environment")
        binding = DailyInstance.read(args.home, selected or next(iter(metadata["environments"])))
    else:
        binding = DailyInstance.read(args.home, selected)
    if args.component in {"code", "manual"}:
        os.environ["GESTALT_DAILY_ENVIRONMENT"] = binding.environment
    os.environ.update(DAGSTER_HOME=str(binding.home), PYTHONPATH=str(code / "src"),
                      PYTHONDONTWRITEBYTECODE="1")
    os.chdir(code)
    if args.component == "manual":
        if not args.job or not args.request:
            p.error("A manual run requires --job and --request")
        result = execute_manual(binding.home, args.job, read_object(args.request))
        print(json.dumps({"run_id": result.run_id, "success": result.success,
                          "environment": binding.environment, "history": str(binding.home)}))
        raise SystemExit(0 if result.success else 1)
    binaries = Path(sys.executable).parent
    workspace = str(binding.home / "workspace.yaml")
    commands = {
        "code": [str(binaries / "dagster"), "api", "grpc", "--host", "127.0.0.1",
                 "--port", str(args.grpc_port), "--module-name",
                 "nba_lineup_model.orchestration.partition_definitions",
                 "--working-directory", str(code), "--use-python-environment-entry-point"],
        "daemon": [str(binaries / "dagster-daemon"), "run", "--workspace", workspace],
        "web": [str(binaries / "dagster-webserver"), "--host", "127.0.0.1",
                "--port", str(args.web_port), "--workspace", workspace],
    }
    command = commands[args.component]
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
