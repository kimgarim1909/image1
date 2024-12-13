#분류 결과 + 이미지 + 텍스트와 함께 분류 결과에 따라 다른 출력 보여주기
import streamlit as st
from fastai.vision.all import *
from PIL import Image
import gdown

# Google Drive 파일 ID
file_id = '1q4nGymjwRWJLl2HePuchabuudk3nq0AZ'

# Google Drive에서 파일 다운로드 함수
@st.cache(allow_output_mutation=True)
def load_model_from_drive(file_id):
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'model.pkl'
    gdown.download(url, output, quiet=False)

    # Fastai 모델 로드
    learner = load_learner(output)
    return learner

def display_left_content(image, prediction, probs, labels):
    st.write("### 왼쪽: 기존 출력 결과")
    if image is not None:
        st.image(image, caption="업로드된 이미지", use_column_width=True)
    st.write(f"예측된 클래스: {prediction}")
    st.markdown("<h4>클래스별 확률:</h4>", unsafe_allow_html=True)
    for label, prob in zip(labels, probs):
        st.markdown(f"""
            <div style="background-color: #f0f0f0; border-radius: 5px; padding: 5px; margin: 5px 0;">
                <strong style="color: #333;">{label}:</strong>
                <div style="background-color: #d3d3d3; border-radius: 5px; width: 100%; padding: 2px;">
                    <div style="background-color: #4CAF50; width: {prob*100}%; padding: 5px 0; border-radius: 5px; text-align: center; color: white;">
                        {prob:.4f}
                    </div>
                </div>
        """, unsafe_allow_html=True)

def display_right_content(prediction, data):
    st.write("### 오른쪽: 동적 분류 결과")
    cols = st.columns(3)

    # 1st Row - Images
    for i in range(3):
        with cols[i]:
            st.image(data['images'][i], caption=f"이미지: {prediction}", use_column_width=True)
    # 2nd Row - YouTube Videos
    for i in range(3):
        with cols[i]:
            st.video(data['videos'][i])
            st.caption(f"유튜브: {prediction}")
    # 3rd Row - Text
    for i in range(3):
        with cols[i]:
            st.write(data['texts'][i])

# 모델 로드
st.write("모델을 로드 중입니다. 잠시만 기다려주세요...")
learner = load_model_from_drive(file_id)
st.success("모델이 성공적으로 로드되었습니다!")

labels = learner.dls.vocab

# 스타일링을 통해 페이지 마진 줄이기
st.markdown("""
    <style>
    .reportview-container .main .block-container {
        max-width: 90%;
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# 분류에 따라 다른 콘텐츠 관리
content_data = {
    labels[0]: {
        'images': [
            "https://i.ibb.co/WP9dbsR/1.jpg",
            "https://i.ibb.co/tcGRqPS/2.jpg",
            "https://i.ibb.co/Rz5d1W9/3.jpg"
        ],
        'videos': [
            "https://youtu.be/pZNnMMPTwm0?feature=shared",
            "https://youtu.be/0-xyHOTnhdU?feature=shared",
            "https://youtu.be/pFvnDf0-v0Y?feature=shared"
        ],
        'texts': [
            "장호해수욕장-강원 삼척시: 스노클링 명소
",
            "을왕리해수욕장-인천 중구: 도심에서 가장 가깝고 아름다운 해수욕장",
            "장등해변-전남 여수시: 느긋하고 조용하게 피서를 즐길 수 있는 해수욕장"
        ]
    },
    labels[1]: {
        'images': [
            "https://i.ibb.co/S5dd1Dz/ef27b324-ef1a-4d68-957f-0012decd642d.jpg",
            "https://i.ibb.co/VCB33Xt/2.jpg",
            "https://i.ibb.co/DLqyskt/3.jpg"
        ],
        'videos': [
            "https://youtu.be/vhWjG_VUIZE?feature=shared",
            "https://youtu.be/vqVsbj4u-ZM?feature=shared",
            "https://youtu.be/RRyw3AnKmuY?feature=shared"
        ],
        'texts': [
            "민둥산-강원 정선: 가을철 대표 억새 산행지 ",
            "달마산-전남 해남군: 한반도의 끝자락, 백두대간을 마무리하는 산",
            "문수산-경기 김포시: 사계절 경치가 아름다워 김포의 금강산이라 불린다"
        ]
    },
    labels[2]: {
        'images': [
            "https://i.ibb.co/VBP3zMs/1.jpg",
            "https://i.ibb.co/LR130hR/2.jpg",
            "https://i.ibb.co/pj9pCQB/3.jpg"
        ],
        'videos': [
            "https://youtu.be/LaOkllGcZ-c?feature=shared",
            "https://youtu.be/D-UJ0JahnSE?feature=shared",
            "https://youtu.be/pOklPsf5qVo?feature=shared"
        ],
        'texts': [
            "완도 청해진 유적-전남 완도군: 동북아 해상무역왕 장보고의 본거지",
            "강릉 경포대-강원 강릉시: 다섯 개의 달이 뜨는 낭만적인 달맞이 명소",
            "병산서원-경북 안동: 역사적 가치가 있는 고려 중기 교육기관"
        ]
    }
}

# 레이아웃 설정
left_column, right_column = st.columns([1, 2])  # 왼쪽과 오른쪽의 비율 조정

# 파일 업로드 컴포넌트 (jpg, png, jpeg, webp, tiff 지원)
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "png", "jpeg", "webp", "tiff"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = PILImage.create(uploaded_file)
    prediction, _, probs = learner.predict(img)

    with left_column:
        display_left_content(image, prediction, probs, labels)

    with right_column:
        # 분류 결과에 따른 콘텐츠 선택
        data = content_data.get(prediction, {
            'images': ["https://via.placeholder.com/300"] * 3,
            'videos': ["https://www.youtube.com/watch?v=3JZ_D3ELwOQ"] * 3,
            'texts': ["기본 텍스트"] * 3
        })
        display_right_content(prediction, data)

