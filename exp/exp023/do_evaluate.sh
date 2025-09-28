EXP_NAME=exp022
DIR_NAME=outputs/exp022_use_deepseek/20250928080224

for d in $DIR_NAME/checkpoint/*/; do
    uv run python exp/$EXP_NAME/evaluate.py --dir "$d"
done