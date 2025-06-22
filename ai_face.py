import os
from PIL import Image as PILImage
from agno.agent import Agent
from agno.models.google import Gemini
import streamlit as st
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.media import Image as AgnoImage
import cv2
import numpy as np

if "GOOGLE_API_KEY" not in st.session_state:
    st.session_state.GOOGLE_API_KEY = None

with st.sidebar:
    st.title("ℹ️ 配置设置")
    
    if not st.session_state.GOOGLE_API_KEY:
        api_key = st.text_input(
            "请输入您的 Google API 密钥：",
            value="AIzaSyBRRd4eP_iImZVFBDpj3NAk_aIlOzBjnJI",
            type="password"
        )
        st.caption(
            "从 [Google AI Studio](https://aistudio.google.com/app/apikey) 获取您的 API 密钥 🔑"
        )
        if api_key:
            st.session_state.GOOGLE_API_KEY = api_key
            st.success("API 密钥已保存！")
            st.rerun()
    else:
        st.success("API 密钥已配置")
        if st.button("🔄 重置 API 密钥"):
            st.session_state.GOOGLE_API_KEY = None
            st.rerun()
    
    st.markdown("---")
    st.info(
        "这是一个基于人工智能的面部特征分析工具，使用先进的计算机视觉技术和传统面相学知识。"
    )
    st.warning(
        "⚠️免责声明：此工具仅供娱乐和文化了解使用。面相分析结果不应作为判断个人能力、性格或命运的绝对依据。"
    )

# 使用 Gemini 模型
face_agent = Agent(
    model=Gemini(
        id="gemini-2.0-flash",
        api_key=st.session_state.GOOGLE_API_KEY
    ),
    tools=[DuckDuckGoTools()],
    markdown=True
) if st.session_state.GOOGLE_API_KEY else None

if not face_agent:
    st.warning("请在侧边栏配置您的 API 密钥以继续使用")

# 面部检测函数
def detect_faces(image_array):
    """检测图像中的面部"""
    try:
        # 转换为灰度图像
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        
        # 使用OpenCV的人脸检测器
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        return len(faces) > 0, faces
    except Exception as e:
        return False, []

# 面部特征分析查询（增强版，包含性别识别和精准评分）
face_analysis_query = """
你是一位专业的面相学专家、颜值评估专家和年龄评估专家，精通中国传统麻衣相法。请仔细分析这张面部图像，并按以下结构提供详细分析：

### 1. 面部检测确认
- 确认图像中包含清晰的人脸
- 评估图像质量和分析可行性
- 说明面部角度和光照条件

### 2. 性别识别（重要：必须准确识别）
- **性别判断**：仔细观察以下特征来准确判断性别：
  * 面部轮廓：男性通常更方正，女性更圆润
  * 眉毛：男性通常更粗浓，女性更细致
  * 眼部：男性眼窝更深，女性眼部更柔和
  * 鼻子：男性鼻梁更挺直，鼻翼更宽
  * 嘴唇：男性通常更薄，女性更饱满
  * 下巴：男性更方正，女性更尖细
  * 喉结：男性可能有明显喉结
  * 皮肤质感：男性通常更粗糙，女性更细腻
- **置信度**：说明性别判断的可靠程度（高/中/低）
- **关键识别特征**：列出最明显的性别特征

### 3. 颜值评分（满分10分，按性别标准评分）
- **性别化评分标准**：
  * 男性评分标准：阳刚之美、轮廓立体感、眉眼神韵、整体气质
  * 女性评分标准：柔美气质、五官精致度、皮肤状态、整体协调性
- **综合颜值评分**：基于对应性别的审美标准给出客观评分
- **详细评分依据**：
  * 五官比例协调性（2分）
  * 面部对称性（2分）
  * 皮肤质感和状态（2分）
  * 眼部魅力度（2分）
  * 整体气质印象（2分）
- **根据评分来贴标签**
  * 男性标签示例："分数高低可以区分成类似，超级大帅哥，校草，邻居大哥，中年大叔，大土豆等等，可以自由发挥"
  * 女性标签示例："分数高低可以区分成类似，世界小姐，女明星，邻家小妹，广场舞大妈等，可以自由发挥"   
- **性别化俏皮点评**：
  * 男性点评示例："眉宇间有股英气，像年轻版的吴彦祖"、"下巴线条很man，但眼神温柔得像小奶狗"
  * 女性点评示例："眼睛水汪汪的像小鹿斑比"、"笑起来有梨涡，甜得像棉花糖"
- **评分说明**：详细解释各项评分理由，语言要生动有趣
- **提升建议**：给出符合性别特点的美容或形象提升建议

### 4. 精确年龄估算（±3岁范围，考虑性别差异）
- **性别化年龄特征分析**：
  * 男性特征：胡须生长情况、面部线条成熟度、眼角细纹
  * 女性特征：皮肤细腻程度、法令纹深浅、眼部状态
- **估算年龄**：给出具体年龄范围（如：25-28岁）
- **关键特征分析**：
  * 眼部细纹和眼袋状况
  * 法令纹深浅程度
  * 皮肤弹性和光泽度
  * 面部轮廓紧致度
  * 颈部皮肤状态
- **置信度评估**：说明年龄判断的可靠程度
- **性别化俏皮年龄点评**：
  * 男性："看起来像个25岁的成熟大叔"、"外表30岁，内心还是18岁的大男孩"
  * 女性："看起来像个20岁的小仙女"、"外表25岁，眼神却有着少女的纯真"

### 5. 面相特征详细分析（按性别特点分析）
- **五官比例分析**：
  * 眉毛：形状、浓淡、长短、高低（男性关注英气，女性关注秀气）
  * 眼睛：大小、形状、间距、神韵（男性关注深邃，女性关注明亮）
  * 鼻子：高低、大小、鼻翼、鼻梁（男性关注挺拔，女性关注精致）
  * 嘴唇：厚薄、形状、嘴角走向（男性关注线条，女性关注饱满）
  * 下巴：尖圆、长短、双下巴情况（男性关注方正，女性关注尖细）

- **面部结构特征**：
  * 脸型分类（圆形、方形、椭圆形、瓜子脸、国字脸等）
  * 颧骨高低和突出程度
  * 额头宽窄和饱满度
  * 下颌线条和轮廓
  * 面部三庭五眼比例

### 6. 中国传统麻衣相法解读
- **十二宫位分析**：
  * 命宫（印堂）：事业运势和性格特征
  * 财帛宫（鼻头）：财运和理财能力
  * 兄弟宫（眉毛）：兄弟姐妹关系和友情
  * 夫妻宫（眼尾）：婚姻和感情运势
  * 子女宫（眼下）：子女缘分和教育能力
  * 疾厄宫（山根）：健康状况和体质
  * 迁移宫（额角）：出行和变动运势
  * 奴仆宫（下巴）：下属关系和晚年运
  * 官禄宫（额头中央）：事业和官运
  * 田宅宫（眼皮）：房产和家庭运势
  * 福德宫（眉上）：福气和精神状态
  * 父母宫（额头两侧）：父母关系和长辈缘

- **五行面相分析**：
  * 金形人：方脸、皮肤白净，性格刚毅
  * 木形人：长脸、皮肤青白，性格温和
  * 水形人：圆脸、皮肤黑润，性格聪慧
  * 火形人：尖脸、皮肤红润，性格急躁
  * 土形人：方圆脸、皮肤黄润，性格厚重

- **麻衣相法特色解读**：
  * 三停分析：上停（额头）主早年运，中停（眉眼鼻）主中年运，下停（嘴下巴）主晚年运
  * 五岳朝归：额为南岳、鼻为中岳、左颧为东岳、右颧为西岳、下巴为北岳
  * 六府充盈：两颊、两颧、两腮的饱满程度
  * 流年运势：根据面部不同部位推算各年龄段运势

### 7. 传统面相性格解读（按性别特点解读）
- **性格特征分析**：根据五官特征分析性格倾向（考虑性别差异）
- **人际关系**：分析社交能力和人缘
- **事业倾向**：适合的职业方向和发展建议（考虑性别优势）
- **感情特质**：恋爱和婚姻中的表现特点

### 8. 现代科学视角补充
- **心理学角度**：从面部表情和微表情分析心理状态
- **美学原理**：运用黄金比例等美学理论分析面部美感
- **健康指标**：从面色、气色分析可能的健康状况

**重要分析要求**：
1. 必须首先准确识别性别，所有后续分析都要基于正确的性别标准
2. 颜值评分要严格按照对应性别的审美标准
3. 年龄估算要考虑性别差异（女性通常显年轻，男性成熟较快）
4. 俏皮点评要符合性别特点，避免性别混淆
5. 如果性别识别不确定，要明确说明并提供两种可能的分析

**重要声明**：
1. 颜值评分基于对应性别的大众审美标准
2. 年龄估算基于外观特征，考虑性别差异
3. 面相学属于传统文化范畴，分析结果仅供娱乐和文化了解
4. 不应将面相分析作为判断个人能力、性格或命运的绝对依据
5. 建议以积极正面的心态查看分析结果
6. 俏皮点评纯属娱乐，请勿当真

请用中文回复，语言要生动有趣，既专业又不失幽默感。在分析过程中务必准确识别性别，并基于正确的性别标准进行所有评估。
"""

st.title("🎭 智能面相分析系统")
st.write("上传您的面部照片，让AI为您进行专业的面相特征分析")

# Create containers for better organization
upload_container = st.container()
image_container = st.container()
analysis_container = st.container()

with upload_container:
    uploaded_file = st.file_uploader(
        "上传面部照片",
        type=["jpg", "jpeg", "png"],
        help="支持格式：JPG、JPEG、PNG"
    )

if uploaded_file is not None:
    with image_container:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            image = PILImage.open(uploaded_file)
            width, height = image.size
            aspect_ratio = width / height
            new_width = 500
            new_height = int(new_width / aspect_ratio)
            resized_image = image.resize((new_width, new_height))
            
            st.image(
                resized_image,
                caption="上传的图片",
                use_container_width=True
            )
            
            # 进行面部检测
            image_array = np.array(resized_image)
            has_faces, faces = detect_faces(image_array)
            
            if has_faces:
                st.success(f"🎭 检测到 {len(faces)} 个面部，可以进行分析")
            else:
                st.warning("⚠️ 未检测到清晰的面部，分析结果可能不准确")
            
            analyze_button = st.button(
                "🔍 开始面相分析",
                type="primary",
                use_container_width=True
            )
    
    with analysis_container:
        if analyze_button:
            if not face_agent:
                st.error("❌ 请先在侧边栏配置您的 Google API 密钥")
                st.info("💡 您需要从 Google AI Studio 获取 API 密钥才能使用分析功能")
            else:
                with st.spinner("🔄 正在分析面部特征... 请稍候"):
                    try:
                        temp_path = "temp_resized_image.png"
                        resized_image.save(temp_path)
                        
                        # Create AgnoImage object
                        agno_image = AgnoImage(filepath=temp_path)
                        
                        st.markdown("### 🎭 面部特征分析结果")
                        
                        # Run analysis
                        response = face_agent.run(face_analysis_query, images=[agno_image])
                        st.markdown("---")
                        st.markdown(response.content)
                        st.markdown("---")
                        
                        st.caption(
                            "注意：面相分析基于传统文化观点，仅供娱乐参考，不应作为判断个人能力或命运的依据。"
                        )
                            
                        # 清理临时文件
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                            
                    except Exception as e:
                        st.error(f"分析错误：{e}")
                        st.info("请检查您的网络连接和API密钥配置")
else:
    st.info("👆 请上传面部照片开始分析")
