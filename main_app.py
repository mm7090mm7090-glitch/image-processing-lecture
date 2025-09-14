# main_app.py - نسخة كاملة ومحدثة
import streamlit as st
from PIL import Image
import cv2
import numpy as np

st.set_page_config(page_title="سلسلة محاضرات معالجة الصور", layout="wide")
st.title("سلسلة محاضرات معالجة الصور التفاعلية")

# رفع أو استخدام صورة افتراضية
def load_image():
    uploaded_file = st.file_uploader("ارفع صورة:", type=["jpg","png","jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file)
    else:
        img = Image.open("images/default.jpg")
    return img

# تحويل PIL إلى OpenCV
def pil_to_cv(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# تحويل OpenCV إلى PIL (يدعم Gray وBinary)
def cv_to_pil(img):
    if len(img.shape) == 2:  # Gray أو Binary
        return Image.fromarray(img)
    else:  # BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img)

# اختيار المحاضرة
module_option = st.sidebar.selectbox(
    "اختر المحاضرة:",
    [
        "المحاضرة 1: مدخل ومعمارية الصور الرقمية",
        "المحاضرة 2: أنظمة الألوان",
        "المحاضرة 3: العمليات على البكسل",
        "المحاضرة 4: الفلاتر والالتفاف",
        "المحاضرة 5: إزالة الضوضاء",
        "المحاضرة 6: كشف الحواف",
        "المحاضرة 7: العمليات المورفولوجية",
        "المحاضرة 8: التحويلات الهندسية",
        "المحاضرة 9: مشروع ختامي"
    ]
)

# -------------------- المحاضرة 1 --------------------
if module_option.startswith("المحاضرة 1"):
    st.header("المحاضرة 1: مدخل ومعمارية الصور الرقمية")
    st.markdown("""
**النظرية:**  
الصورة الرقمية هي تمثيل رقمي للصور الحقيقية عبر شبكة من البكسلات.  
كل بكسل يحتوي على معلومات اللون والسطوع.  
الأبعاد الأساسية للصورة هي الطول × العرض × عدد القنوات اللونية.  
القنوات الشائعة هي الأحمر والأخضر والأزرق (RGB).  
العمق اللوني (bit depth) يحدد عدد الألوان الممكن تمثيلها لكل بكسل.  
فهم هذه المفاهيم مهم قبل التعامل مع أي معالجة أو تحليل للصور.
""")
    img = load_image()
    st.image(img, caption="الصورة الأصلية", use_container_width=True)
    st.write("أبعاد الصورة:", img.size)
    st.write("عدد القنوات:", len(img.getbands()))

# -------------------- المحاضرة 2 --------------------
elif module_option.startswith("المحاضرة 2"):
    st.header("المحاضرة 2: أنظمة الألوان")
    st.markdown("""
**النظرية:**  
تستخدم أنظمة الألوان لتمثيل المعلومات اللونية في الصورة.  
RGB هو النظام الأكثر شيوعًا حيث يمثل كل قناة لون أساسي.  
Gray يمثل الصورة بدرجات الرمادي فقط، مناسب لتقليل التعقيد الحسابي.  
HSV يفصل التدرج (Hue) عن التشبع (Saturation) والإضاءة (Value)، مفيد للتحليل اللوني.  
يمكن تقسيم كل نظام إلى قنوات منفصلة لمزيد من المعالجة.  
اختيار النظام الصحيح يعتمد على التطبيق المطلوب والنتائج المرغوبة.
""")
    img = load_image()
    img_cv = pil_to_cv(img)

    color_space = st.selectbox("اختر نظام الألوان:", ["RGB","Gray","HSV"])
    if color_space=="Gray":
        img_processed = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        st.image(cv_to_pil(img_processed), caption="Gray", use_container_width=True)
    elif color_space=="HSV":
        img_processed = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        st.image(cv_to_pil(img_processed), caption="HSV", use_container_width=True)
    else:
        st.image(img, caption="RGB", use_container_width=True)

    if st.checkbox("عرض القنوات R/G/B"):
        b,g,r = cv2.split(img_cv)
        st.image(cv_to_pil(cv2.merge([r,np.zeros_like(r),np.zeros_like(r)])), caption="R", use_container_width=True)
        st.image(cv_to_pil(cv2.merge([np.zeros_like(g),g,np.zeros_like(g)])), caption="G", use_container_width=True)
        st.image(cv_to_pil(cv2.merge([np.zeros_like(b),np.zeros_like(b),b])), caption="B", use_container_width=True)

# -------------------- المحاضرة 3 --------------------
elif module_option.startswith("المحاضرة 3"):
    st.header("المحاضرة 3: العمليات على البكسل")
    st.markdown("""
**النظرية:**  
العمليات على البكسل تتحكم في السطوع والتباين والتمثيل العددي للصور.  
تعديل السطوع يزيد أو يقلل من إضاءة الصورة بشكل متساوي.  
التباين يحدد مدى اختلاف الألوان بين البكسلات.  
الصور السالبة (Negative) تعكس قيم البكسلات للحصول على تأثير بصري مختلف.  
Thresholding يحول الصورة إلى ثنائية لتسهيل التحليل.  
هذه العمليات أساسية للتحضير لأي خطوات معالجة متقدمة.
""")
    img = load_image()
    img_cv = pil_to_cv(img)

    brightness = st.slider("تحكم بالسطوع", -100, 100, 0)
    contrast = st.slider("تحكم بالتباين", 0.5, 3.0, 1.0)
    img_bright = cv2.convertScaleAbs(img_cv, alpha=contrast, beta=brightness)
    st.image(cv_to_pil(img_bright), caption="بعد تعديل السطوع/التباين", use_container_width=True)

    if st.button("تطبيق Negative"):
        img_neg = 255 - img_cv
        st.image(cv_to_pil(img_neg), caption="Negative", use_container_width=True)

    threshold_val = st.slider("Threshold", 0, 255, 127)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, thresh_img = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)
    st.image(cv_to_pil(thresh_img), caption="Thresholding", use_container_width=True)

# -------------------- المحاضرة 4 --------------------
elif module_option.startswith("المحاضرة 4"):
    st.header("المحاضرة 4: الفلاتر والالتفاف")
    st.markdown("""
**النظرية:**  
الفلاتر تُستخدم لتغيير خصائص الصورة مثل الحدة أو التنعيم.  
Kernel أو الماسك هو مصفوفة تحدد كيفية تأثير البكسلات على بعضها البعض.  
Sharpen يزيد وضوح التفاصيل ويبرز الحواف.  
Blur يقلل التشويش ويعطي تأثير تمويه.  
Edge detection يساعد على تحديد حدود الأجسام داخل الصورة.  
الفهم الجيد للفلاتر والماسك ضروري لتحكم كامل في مخرجات الصورة.
""")
    img = load_image()
    img_cv = pil_to_cv(img)

    filter_type = st.selectbox("اختر الفلتر:", ["Blur","Sharpen","Edge"])
    k_size = st.slider("حجم Kernel", 1, 15, 3, step=2)

    if filter_type=="Blur":
        img_filtered = cv2.GaussianBlur(img_cv,(k_size,k_size),0)
    elif filter_type=="Sharpen":
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        img_filtered = cv2.filter2D(img_cv,-1,kernel)
    else:  # Edge
        img_filtered = cv2.Canny(cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY),50,150)
    st.image(cv_to_pil(img_filtered) if filter_type!="Edge" else img_filtered, caption="بعد التطبيق", use_container_width=True)

# -------------------- المحاضرة 5 --------------------
elif module_option.startswith("المحاضرة 5"):
    st.header("المحاضرة 5: إزالة الضوضاء")
    st.markdown("""
**النظرية:**  
الصور غالبًا تحتوي على ضوضاء تؤثر على الجودة والتحليل.  
الضوضاء الشائعة تشمل Salt & Pepper وGaussian Noise.  
Median Filter يزيل الضوضاء النقطية الصغيرة بدون فقدان التفاصيل.  
Bilateral Filter يقلل الضوضاء مع الحفاظ على الحواف.  
إزالة الضوضاء خطوة مهمة قبل أي معالجة أو تحليل متقدم.  
اختيار الفلتر المناسب يعتمد على نوع وشدة الضوضاء في الصورة.
""")
    img = load_image()
    img_cv = pil_to_cv(img)

    noise_type = st.selectbox("إضافة ضوضاء:", ["None","Salt & Pepper","Gaussian"])
    img_noisy = img_cv.copy()
    if noise_type=="Salt & Pepper":
        s_vs_p = 0.5
        amount = 0.04
        out = img_noisy.copy()
        num_salt = np.ceil(amount * img_noisy.size * s_vs_p)
        coords = [np.random.randint(0, i-1, int(num_salt)) for i in img_noisy.shape]
        out[coords[0],coords[1],:] = 255
        num_pepper = np.ceil(amount * img_noisy.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i-1, int(num_pepper)) for i in img_noisy.shape]
        out[coords[0],coords[1],:] = 0
        img_noisy = out
    elif noise_type=="Gaussian":
        row,col,ch = img_noisy.shape
        mean = 0
        var = 10
        sigma = var ** 0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        img_noisy = cv2.add(img_noisy.astype(np.float32), gauss.astype(np.float32))
        img_noisy = np.clip(img_noisy,0,255).astype(np.uint8)

    st.image(cv_to_pil(img_noisy), caption="بعد إضافة الضوضاء", use_container_width=True)

    denoise_type = st.selectbox("اختر فلتر إزالة الضوضاء:", ["Median","Bilateral"])
    if denoise_type=="Median":
        img_denoised = cv2.medianBlur(img_noisy,5)
    else:
        img_denoised = cv2.bilateralFilter(img_noisy,9,75,75)
    st.image(cv_to_pil(img_denoised), caption="بعد إزالة الضوضاء", use_container_width=True)

# -------------------- المحاضرة 6 --------------------
elif module_option.startswith("المحاضرة 6"):
    st.header("المحاضرة 6: كشف الحواف")
    st.markdown("""
**النظرية:**  
كشف الحواف يساعد على تحديد حدود الأجسام في الصورة.  
يعتمد على التغيرات المفاجئة في شدة الإضاءة بين البكسلات المجاورة.  
طرق الكشف تشمل Sobel وLaplacian وCanny.  
Sobel يحسب مشتقات الصورة في الاتجاهين x و y لتحديد التغيرات.  
Laplacian يعتمد على المشتقة الثانية للعثور على مناطق التغير السريع.  
Canny يعطي حواف دقيقة بعد تطبيق التنعيم والعتبات لتقليل الضوضاء.
""")
    img = load_image()
    img_cv = pil_to_cv(img)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    edge_type = st.selectbox("اختر نوع كشف الحافة:", ["Sobel","Laplacian","Canny"])
    if edge_type=="Sobel":
        img_edge = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
        img_edge = cv2.convertScaleAbs(img_edge)
    elif edge_type=="Laplacian":
        img_edge = cv2.Laplacian(gray,cv2.CV_64F)
        img_edge = cv2.convertScaleAbs(img_edge)
    else:
        t1 = st.slider("Threshold1 (Canny)", 0, 255, 100)
        t2 = st.slider("Threshold2 (Canny)", 0, 255, 200)
        img_edge = cv2.Canny(gray,t1,t2)
    st.image(cv_to_pil(img_edge), caption="كشف الحواف", use_container_width=True)

# -------------------- المحاضرة 7 --------------------
elif module_option.startswith("المحاضرة 7"):
    st.header("المحاضرة 7: العمليات المورفولوجية")
    st.markdown("""
**النظرية:**  
العمليات المورفولوجية تُطبق على الصور الثنائية لمعالجة شكل الأجسام.  
Erosion تقلل من حجم الأجسام وتزيل الضوضاء الصغيرة.  
Dilation تكبر الأجسام وتملأ الفراغات الصغيرة.  
Opening هو Erosion متبوعًا بـ Dilation لإزالة الضوضاء الصغيرة.  
Closing هو Dilation متبوعًا بـ Erosion لملء الفجوات الصغيرة.  
تُستخدم هذه العمليات في تحضير الصور قبل التحليل الهندسي أو اكتشاف الكائنات.
""")
    img = load_image()
    img_cv = pil_to_cv(img)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    st.image(cv_to_pil(binary), caption="Binary", use_container_width=True)

    morph_op = st.selectbox("اختر العملية المورفولوجية:", ["Erosion","Dilation","Opening","Closing"])
    k_size = st.slider("حجم Kernel",1,15,3,step=2)
    kernel = np.ones((k_size,k_size),np.uint8)

    if morph_op=="Erosion":
        img_morph = cv2.erode(binary,kernel,iterations=1)
    elif morph_op=="Dilation":
        img_morph = cv2.dilate(binary,kernel,iterations=1)
    elif morph_op=="Opening":
        img_morph = cv2.morphologyEx(binary,cv2.MORPH_OPEN,kernel)
    else:
        img_morph = cv2.morphologyEx(binary,cv2.MORPH_CLOSE,kernel)
    st.image(cv_to_pil(img_morph), caption="بعد العملية المورفولوجية", use_container_width=True)

# -------------------- المحاضرة 8 --------------------
elif module_option.startswith("المحاضرة 8"):
    st.header("المحاضرة 8: التحويلات الهندسية")
    st.markdown("""
**النظرية:**  
التحويلات الهندسية تغير موضع أو حجم أو شكل الصورة.  
Translation تنقل الصورة إلى موقع آخر داخل الإطار.  
Rotation يدور الصورة حول نقطة معينة بزوايا مختلفة.  
Scaling يغير حجم الصورة لتكبيرها أو تصغيرها.  
Flipping يعكس الصورة أفقيًا أو رأسيًا.  
Cropping يقطع جزءًا محددًا من الصورة للتركيز على منطقة معينة.
""")
    img = load_image()
    img_cv = pil_to_cv(img)

    st.subheader("دوران الصورة")
    angle = st.slider("زاوية الدوران", 0, 360, 0)
    (h,w) = img_cv.shape[:2]
    M = cv2.getRotationMatrix2D((w/2,h/2),angle,1)
    rotated = cv2.warpAffine(img_cv,M,(w,h))
    st.image(cv_to_pil(rotated), caption=f"دوران {angle}°", use_container_width=True)

    st.subheader("تكبير/تصغير")
    scale = st.slider("نسبة التكبير/التصغير", 0.1, 3.0, 1.0)
    resized = cv2.resize(img_cv,None,fx=scale,fy=scale)
    st.image(cv_to_pil(resized), caption=f"تكبير/تصغير x{scale}", use_container_width=True)

    st.subheader("انعكاس الصورة")
    flip_type = st.selectbox("نوع الانعكاس", ["لا شيء","أفقي","رأسي"])
    if flip_type=="أفقي":
        flipped = cv2.flip(img_cv,1)
    elif flip_type=="رأسي":
        flipped = cv2.flip(img_cv,0)
    else:
        flipped = img_cv
    st.image(cv_to_pil(flipped), caption=f"انعكاس: {flip_type}", use_container_width=True)

    st.subheader("قص جزء من الصورة")
    x = st.slider("x البداية",0,w-1,0)
    y = st.slider("y البداية",0,h-1,0)
    x2 = st.slider("x النهاية",0,w-1,w-1)
    y2 = st.slider("y النهاية",0,h-1,h-1)
    cropped = img_cv[y:y2, x:x2]
    st.image(cv_to_pil(cropped), caption="المنطقة المقصوصة", use_container_width=True)

# -------------------- المحاضرة 9 --------------------
elif module_option.startswith("المحاضرة 9"):
    st.header("المحاضرة 9: المشروع النهائي")
    st.markdown("""
**النظرية:**  
المشروع النهائي يجمع جميع العمليات السابقة في سلسلة تفاعلية.  
يمكن رفع صورة وتطبيق عمليات مثل Gray, Blur, وEdges بالتتابع.  
يسمح للمستخدم برؤية التغييرات مباشرة قبل وبعد كل عملية.  
الهدف هو فهم كيفية تكامل خطوات المعالجة المختلفة للحصول على النتيجة المطلوبة.  
يمكن حفظ الصورة الناتجة للاستخدام في مشاريع أخرى أو التحليل.  
هذا يعزز الفهم العملي للنظرية ويطور المهارات التطبيقية.
""")
    img = load_image()
    img_cv = pil_to_cv(img)
    st.image(cv_to_pil(img_cv), caption="الصورة الأصلية", use_container_width=True)

    operations = st.multiselect("اختر العمليات لتطبيقها (Pipeline):", 
                                ["Gray","Blur","Edges"])
    result = img_cv.copy()

    if "Gray" in operations:
        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    if "Blur" in operations:
        if len(result.shape)==2:
            result = cv2.GaussianBlur(result,(5,5),0)
        else:
            result = cv2.GaussianBlur(result,(5,5),0)
    if "Edges" in operations:
        if len(result.shape)==3:
            gray_tmp = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        else:
            gray_tmp = result
        result = cv2.Canny(gray_tmp,100,200)

    st.image(cv_to_pil(result), caption="النتيجة النهائية", use_container_width=True)
    if st.button("حفظ الصورة الناتجة"):
        save_path = "images/result.png"
        if len(result.shape)==2:
            cv2.imwrite(save_path,result)
        else:
            cv2.imwrite(save_path,cv2.cvtColor(result,cv2.COLOR_RGB2BGR))
        st.success(f"تم حفظ الصورة في {save_path}")
