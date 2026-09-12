# -*- coding: utf-8 -*-
"""테스트 전역 설정.

tests/test_cli_interactive.py 는 pellet-support CLI 를 서브프로세스로 실행하며
``examples/out/table.stl`` 이 이미 있다고 가정한다. 로컬에서는 개발자가
``python examples/make_test_models.py`` 를 먼저 돌려 본 적이 있어 우연히
존재했지만, CI 는 이 스크립트를 실행하지 않아 매번 "string is not a file" 로
실패했다. 테스트가 스스로 필요한 입력을 준비하게 한다.
"""

import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture(scope="session", autouse=True)
def _example_models():
    sys.path.insert(0, str(EXAMPLES_DIR))
    import make_test_models

    make_test_models.main()
