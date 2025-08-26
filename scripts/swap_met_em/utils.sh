# Function to generate list of hourly timestamps

generate_hours() {
  local start=$1
  local end=$2
  local list=()

  local t=$(date -u -d "${start/_/ }" +%s)
  local end_t=$(date -u -d "${end/_/ }" +%s)

  while [ $t -le $end_t ]; do
    list+=($(date -u -d @$t +%Y-%m-%d_%H:%M:%S))
    t=$((t+3600))
  done
  echo "${list[@]}"
}

