# emotion_chatbot
멀티모달 감정 상담 챗봇

프로젝트명: 감정 인식 기반 개인비서 Agent
목표:

사용자의 얼굴 이미지(또는 영상)를 분석하여 감정을 판별하고, 이에 맞는 응답 및 행동을 추천하는 개인비서 시스템을 구축합니다.
사용자의 개인정보와 취향 정보(텍스트 데이터)를 벡터 DB(FAISS)를 통해 저장 및 검색하여, 보다 맞춤형 추천을 제공합니다.
LangChain과 LangGraph를 활용하여 RAG(Retrieval Augmented Generation) 기반 응답 생성을 구현합니다.
이메일 전송 및 캘린더 파일 생성 같은 사용자 편의 기능도 포함되어 있습니다.
