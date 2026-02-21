from __future__ import annotations

from _train_common import build_parser, run_training_from_configs


def main() -> None:
    parser = build_parser(
        default_model_cfg="configs/model/resnet34_baseline.yaml",
        default_loss_cfg="configs/loss/stage3_finetune.yaml",
        default_train_cfg="configs/train/stage3_finetune.yaml",
        default_init_from="outputs/runs/stage2_hybrid/best.pt",
    )
    args = parser.parse_args()

    metrics = run_training_from_configs(
        model_config_path=args.model_config,
        loss_config_path=args.loss_config,
        train_config_path=args.train_config,
        use_synthetic=bool(args.use_synthetic),
        loader_module=args.loader_module,
        loader_fn=args.loader_fn,
        resume_from=args.resume_from,
        init_from=args.init_from,
        validate_only=bool(args.validate_only),
        run_name_override=args.run_name,
        warp_backend_override=args.warp_backend,
    )
    print("Stage3 metrics:", metrics)


if __name__ == "__main__":
    main()
