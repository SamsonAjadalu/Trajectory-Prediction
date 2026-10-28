"""Check that a running trajectory API returns a valid twelve-step forecast."""

from __future__ import annotations

import argparse
import json
from urllib.request import Request, urlopen


def get_json(url: str) -> dict:
    with urlopen(url, timeout=30) as response:  # noqa: S310 - caller supplies local URL
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    args = parser.parse_args()

    health = get_json(f'{args.url}/health')
    if health.get('status') != 'ok':
        raise RuntimeError(f'unhealthy API response: {health}')

    payload = json.dumps({
        'history': [[0.0, 0.0], [1.0, 0.1], [2.0, 0.2], [3.0, 0.3]],
    }).encode()
    request = Request(
        f'{args.url}/predict',
        data=payload,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    with urlopen(request, timeout=30) as response:  # noqa: S310 - local smoke URL
        prediction = json.load(response)

    trajectory = prediction.get('trajectory')
    if not isinstance(trajectory, list) or len(trajectory) != 12:
        raise RuntimeError(f'expected 12 trajectory steps, got {trajectory!r}')
    if any(not isinstance(step, list) or len(step) != 2 for step in trajectory):
        raise RuntimeError(f'expected [12, 2] trajectory, got {trajectory!r}')
    print('health=ok')
    print('trajectory_shape=[12, 2]')
    print(f"model_uri={prediction.get('model_uri')}")


if __name__ == '__main__':
    main()
