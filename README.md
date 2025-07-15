# Setup
## Setup RLViser
`curl -o rlviser -L https://github.com/VirxEC/rlviser/releases/download/v0.7.14/rlviser`
`chmod +x rlviser`

## Initialize environment
`uv sync`

# Run

## Start training
`uv run train.py exp=offense`

## Watch agent with RLViser
#### This will only work once a checkpoint has been generated
`uv run load_latest.py`

