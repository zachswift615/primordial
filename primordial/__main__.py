"""CLI entry point for Primordial."""

import argparse
import sys


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Primordial - Human Teaching Interface for AI Agents"
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Target frames per second (default: 60)"
    )

    parser.add_argument(
        "--zoom",
        type=float,
        default=1.0,
        help="Initial zoom level (default: 1.0)"
    )

    parser.add_argument(
        "--width",
        type=int,
        default=960,
        help="Window width in pixels (default: 960)"
    )

    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Window height in pixels (default: 720)"
    )

    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Disable audio capture"
    )

    args = parser.parse_args()

    # Import here to avoid pygame init on import
    from primordial.interface.config import UIConfig
    from primordial.interface.app import TeachingApp

    # Create config from args
    config = UIConfig()
    config.fps = args.fps
    config.window_width = args.width
    config.window_height = args.height

    # Create and run app
    app = TeachingApp(config)

    # Disable audio if requested
    if args.no_audio:
        app.audio_capture.stop()

    try:
        app.run()
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        app.stop()
        sys.exit(0)


if __name__ == "__main__":
    main()
