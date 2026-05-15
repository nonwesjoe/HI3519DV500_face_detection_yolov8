atc --output="./model/yolov8_face" \
    --insert_op_conf=./insert_op.cfg \
    --framework=5 \
    --save_original_model=true \
    --model="./onnx_model/yolov8n-face.onnx" \
    --image_list="images:./data/image_ref_list.txt"
