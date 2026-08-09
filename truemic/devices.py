"""Find audio devices by name substring — indices reshuffle on USB plug."""

import sounddevice as sd


def find_device(substring: str | None, kind: str) -> int | None:
    if substring is None:
        return None  # sounddevice picks the system default
    channel_key = f"max_{kind}_channels"
    for index, device in enumerate(sd.query_devices()):
        if device[channel_key] > 0 and substring.lower() in device["name"].lower():
            return index
    raise LookupError(
        f"No {kind} device matching {substring!r}. "
        "Run `python -m truemic.devices` to list devices."
    )


def main() -> None:
    for index, device in enumerate(sd.query_devices()):
        kinds = []
        if device["max_input_channels"] > 0:
            kinds.append(f"in:{device['max_input_channels']}")
        if device["max_output_channels"] > 0:
            kinds.append(f"out:{device['max_output_channels']}")
        print(f"{index:3d}  {device['name']}  [{' '.join(kinds)}]")


if __name__ == "__main__":
    main()
