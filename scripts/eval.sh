# Possible MODEL_TYPE:
# Prior methods:
# pi05_baseline, MemER

# Symbolic Memory:
# symbolic_simpleSG_oracle,  symbolic_simpleSG_gemini,  symbolic_simpleSG_qwenvl, 
# symbolic_groundedSG_oracle, symbolic_groundedSG_gemini, symbolic_groundedSG_qwenvl

# Perceptual Memory:
# perceptual-framesamp-context, perceptual-framesamp-modul, perceptual-framesamp-expert
# perceptual-tokendrop-context, perceptual-tokendrop-modul, perceptual-tokendrop-expert

# Recurrent Memory:
# recurrent-rmt-context, recurrent-rmt-modul, recurrent-rmt-expert
# recurrent-ttt-context, recurrent-ttt-modul, recurrent-ttt-expert

# set the MODEL_TYPE from the list above according to your needs

#### set your own parameters ####
MODEL_TYPE="symbolic_groundedSG_qwenvl"
SEED=7          # model seed for evaluation; change this to use different seeds for multiple runs
CKPT_ID=79999   # ckpt id for evaluation; change this to use different checkpoints
GPU_ID_server=1 # gpu id for server; when set, the VLA policy server will run on this GPU
GPU_ID_client=0 # gpu id for client; when set, the RoboMME environment and/or VLM subgoal predictor will run on this GPU
#--------------------------------#



find_free_port() {
  local min=${1:-2000}
  local max=${2:-30000}
  local port
  local tries=5000  # max tries to find a free port

  for ((i=0; i<tries; i++)); do
    port=$(shuf -i"${min}"-"${max}" -n1)
    if ! lsof -iTCP:"${port}" -sTCP:LISTEN &>/dev/null; then
      echo "${port}"
      return 0
    fi
  done

  echo "ERROR: not found free port in range ${min}-${max}" >&2
  return 1
}
PORT=$(find_free_port)


if [ "$MODEL_TYPE" == "pi05_baseline" ]; then
    CONFIG_TYPE="pi05_baseline"
    EXTRA_ARGS="--args.no-use-history"
else
    CONFIG_TYPE="mme_vla_suite"
    # symbolic_memory
    if [ "$MODEL_TYPE" == "symbolic_simpleSG_oracle" ]; then
        EXTRA_ARGS="--args.use-oracle --args.subgoal-type=simple_subgoal"
        MODEL_TYPE="symbolic-simple-subgoal"
    elif [ "$MODEL_TYPE" == "symbolic_groundedSG_oracle" ]; then
        EXTRA_ARGS="--args.use-oracle --args.subgoal-type=grounded_subgoal"
        MODEL_TYPE="symbolic-grounded-subgoal"

    elif [ "$MODEL_TYPE" == "symbolic_simpleSG_qwenvl" ]; then
        EXTRA_ARGS="--args.use-qwenvl --args.subgoal-type=simple_subgoal"
        MODEL_TYPE="symbolic-simple-subgoal"
    elif [ "$MODEL_TYPE" == "symbolic_groundedSG_qwenvl" ]; then
        EXTRA_ARGS="--args.use-qwenvl --args.subgoal-type=grounded_subgoal"
        MODEL_TYPE="symbolic-grounded-subgoal"

    elif [ "$MODEL_TYPE" == "symbolic_simpleSG_gemini" ]; then
        EXTRA_ARGS="--args.use-gemini --args.subgoal-type=simple_subgoal"
        MODEL_TYPE="symbolic-simple-subgoal"
    elif [ "$MODEL_TYPE" == "symbolic_groundedSG_gemini" ]; then
        EXTRA_ARGS="--args.use-gemini --args.subgoal-type=grounded_subgoal"
        MODEL_TYPE="symbolic-grounded-subgoal"

    elif [ "$MODEL_TYPE" == "MemER" ]; then
        EXTRA_ARGS="--args.use-memer --args.subgoal-type=grounded_subgoal"
        MODEL_TYPE="symbolic-grounded-subgoal" # we use grounded subgoal for MemER in our experiments
    
    # perceptual_memory or recurrent_memory
    else
        EXTRA_ARGS=""
    fi
fi


session_name="${MODEL_TYPE}_ckpt${CKPT_ID}_seed${SEED}_port${PORT}"
echo "Evaluating $MODEL_TYPE with seed $SEED and ckpt id $CKPT_ID on port $PORT"


# Check if tmux session already exists
tmux has-session -t $session_name 2>/dev/null

if [ $? != 0 ]; then
    # Create new tmux session with first window for serve_policy
    tmux new-session -d -s $session_name -n "serve_policy" 
    tmux send-keys -t $session_name:serve_policy "CUDA_VISIBLE_DEVICES=$GPU_ID_server uv run scripts/serve_policy.py --seed=$SEED  --port=$PORT policy:checkpoint --policy.dir=runs/ckpts/$CONFIG_TYPE/$MODEL_TYPE/$CKPT_ID --policy.config=$CONFIG_TYPE" Enter

    sleep 30
    
    # Create second window for eval in the same session
    tmux new-window -t $session_name -n "eval"
    tmux send-keys -t $session_name:eval "micromamba activate robomme" Enter
    tmux send-keys -t $session_name:eval "CUDA_VISIBLE_DEVICES=$GPU_ID_client python examples/robomme/eval.py --args.model_seed=$SEED --args.port=$PORT --args.policy_name=$MODEL_TYPE --args.model_ckpt_id=$CKPT_ID ${EXTRA_ARGS}; tmux wait-for -S eval-done" Enter

    # Wait for eval to complete, or exit if tmux session is killed
    tmux wait-for eval-done &
    wait_pid=$!
    while kill -0 $wait_pid 2>/dev/null; do
        tmux has-session -t $session_name 2>/dev/null || { kill $wait_pid 2>/dev/null; echo "Tmux session killed, exiting."; exit 1; }
        sleep 2
    done
    tmux kill-session -t $session_name 2>/dev/null || true

else
    echo "Tmux session ${session_name} already exists. Change the port or use a different session."
fi