from flask import Flask, request, redirect, url_for, render_template, Markup
from werkzeug.utils import secure_filename
import tensorflow as tf

import os
import shutil
from PIL import Image
import numpy as np

UPLOAD_FOLDER = "./static/images/"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

labels = ["歯肉炎ではありません", "軽度の歯肉炎です", "中から重度の歯肉炎です"]
n_class = len(labels)
img_size = 224
n_result = 1 # 上位3つの結果を表示

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")


@app.route("/result", methods=["GET", "POST"])
def result():
    if request.method == "POST":
        # ファイルの存在と形式を確認
        if "file" not in request.files:
            print("File doesn't exist!")
            return redirect(url_for("index"))
        file = request.files["file"]
        if not allowed_file(file.filename):
            print(file.filename + ": File not allowed!")
            return redirect(url_for("index"))

        # ファイルの保存
        if os.path.isdir(UPLOAD_FOLDER):
            shutil.rmtree(UPLOAD_FOLDER)
        os.mkdir(UPLOAD_FOLDER)
        filename = secure_filename(file.filename)  # ファイル名を安全なものに
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # 画像の読み込み
        image = Image.open(filepath)
        image = image.convert("RGB")
        image = image.resize((img_size, img_size))
        image = np.array(image)
        image = image.astype('float') / 255.0
        image = tf.reshape(image, [1, 224, 224, 3])

        # 予測
        new_model = tf.keras.models.load_model('../GingivitisEstimator_streamlit/my_model_EN_adam.h5')
        pred = new_model.predict(image)
        sorted_idx = np.argsort(-pred[0])  # 降順でソート
        result = ""
        for i in range(n_result):
            idx = sorted_idx[i]
            ratio = pred[0][idx]
            label = labels[idx]
            result += "<p>" +  label + "</p>"
        return render_template("result.html", result=Markup(result), filepath=filepath)
    else:
        return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
