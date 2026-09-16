# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.serve.fault_tolerance.api_router import (
    register_fault_tolerance_api_router,
)


@pytest.mark.parametrize("request_id", [None, "", "   "])
def test_apply_rejects_missing_or_blank_request_id(request_id):
    """A blank ID would match the status left by an earlier recovery round."""
    app = FastAPI()
    register_fault_tolerance_api_router(app)
    body = {"instruction": "retry", "params": {}}
    if request_id is not None:
        body["request_id"] = request_id

    response = TestClient(app).post("/v1/fault_tolerance/apply", json=body)

    assert response.status_code == 400
    assert response.json()["detail"] == "'request_id' must be a non-empty string."
