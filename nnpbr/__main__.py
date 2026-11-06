import sys

if sys.argv[1:2] and sys.argv[1] in ("smoke", "train", "view", "check"):
    from .material import main
    main(sys.argv[1:])
else:
    from .cli import main
    main()
