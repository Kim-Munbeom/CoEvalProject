"""
CoEval V2 Configuration Module

가중치 설정 및 검증을 담당하는 모듈입니다.
평가 기준별 가중치를 관리하고, 합계가 1.0이 되도록 검증합니다.
"""

import os
from pydantic import BaseModel, field_validator
from typing import Dict


class WeightsConfig(BaseModel):
    """평가 가중치 설정

    3개 평가 기준의 가중치를 관리합니다:
    - actionability: 실행가능성 (정확성, 명료성, 관련성, 완전성)
    - expertise: 전문성 (구체 정보, 실무 디테일)
    - practicality: 현실성 (멘티 상황 고려, 경험 기반 조언)

    기본값:
    - 실행가능성: 40% (0.4)
    - 전문성: 30% (0.3)
    - 현실성: 30% (0.3)
    """

    actionability: float = 0.4
    expertise: float = 0.3
    practicality: float = 0.3

    @field_validator('actionability', 'expertise', 'practicality')
    @classmethod
    def check_range(cls, v: float) -> float:
        """가중치가 0~1 범위 내에 있는지 검증

        Args:
            v: 가중치 값

        Returns:
            float: 검증된 가중치 값

        Raises:
            ValueError: 가중치가 0~1 범위를 벗어난 경우
        """
        if not 0 <= v <= 1:
            raise ValueError(f"가중치는 0~1 사이여야 합니다 (입력값: {v})")
        return v

    def validate_sum(self) -> bool:
        """가중치 합계가 1.0인지 검증

        Returns:
            bool: 검증 성공 시 True

        Raises:
            ValueError: 가중치 합계가 1.0이 아닌 경우
        """
        total = self.actionability + self.expertise + self.practicality
        if abs(total - 1.0) > 0.001:  # 부동소수점 오차 허용
            raise ValueError(
                f"가중치 합은 1.0이어야 합니다 (현재: {total:.3f})\n"
                f"  - 실행가능성: {self.actionability}\n"
                f"  - 전문성: {self.expertise}\n"
                f"  - 현실성: {self.practicality}"
            )
        return True

    @classmethod
    def from_env(cls) -> "WeightsConfig":
        """환경 변수에서 가중치 로드

        환경 변수:
        - WEIGHT_ACTIONABILITY: 실행가능성 가중치 (기본값: 0.4)
        - WEIGHT_EXPERTISE: 전문성 가중치 (기본값: 0.3)
        - WEIGHT_PRACTICALITY: 현실성 가중치 (기본값: 0.3)

        Returns:
            WeightsConfig: 환경 변수 또는 기본값으로 초기화된 설정
        """
        return cls(
            actionability=float(os.getenv("WEIGHT_ACTIONABILITY", "0.4")),
            expertise=float(os.getenv("WEIGHT_EXPERTISE", "0.3")),
            practicality=float(os.getenv("WEIGHT_PRACTICALITY", "0.3"))
        )

    def to_dict(self) -> Dict[str, float]:
        """가중치를 딕셔너리로 변환

        Returns:
            Dict[str, float]: 가중치 딕셔너리
        """
        return {
            "actionability": self.actionability,
            "expertise": self.expertise,
            "practicality": self.practicality
        }

    def to_percentage_dict(self) -> Dict[str, int]:
        """가중치를 퍼센트(0-100) 딕셔너리로 변환

        Returns:
            Dict[str, int]: 가중치 퍼센트 딕셔너리
        """
        return {
            "actionability": int(self.actionability * 100),
            "expertise": int(self.expertise * 100),
            "practicality": int(self.practicality * 100)
        }

    def __str__(self) -> str:
        """가중치를 읽기 쉬운 문자열로 변환"""
        return (
            f"WeightsConfig(\n"
            f"  실행가능성: {self.actionability*100:.0f}%\n"
            f"  전문성: {self.expertise*100:.0f}%\n"
            f"  현실성: {self.practicality*100:.0f}%\n"
            f")"
        )


# 기본 가중치 설정 (전역)
DEFAULT_WEIGHTS = WeightsConfig()
