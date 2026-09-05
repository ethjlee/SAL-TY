#!/usr/bin/env bash
# Source this file to use watch_scrapers, or run it directly with bash.
watch_scrapers() (
    local once=false interval="${WATCH_INTERVAL:-30}"
    if [[ "${1:-}" == --once ]]; then
        once=true
        shift
    fi
    local -a containers=("$@")
    # Container names from compose.yml; pass names to select a subset.
    ((${#containers[@]})) || containers=(salty{1..11})
    if [[ ! "$interval" =~ ^[0-9]+([.][0-9]+)?$ || ! "$interval" =~ [1-9] ]]; then
        printf 'WATCH_INTERVAL must be a positive number of seconds.\n' >&2
        return 1
    fi
    command -v docker >/dev/null || { printf 'docker is required.\n' >&2; return 1; }
    trap 'exit 0' INT TERM

    local container state logs progress
    while :; do
        if [[ -t 1 && "$once" == false ]]; then
            printf '\033[H\033[2J'
        fi
        printf 'Scraper progress — %s (Ctrl-C to stop)\n\n' "$(date '+%H:%M:%S')"
        for container in "${containers[@]}"; do
            if ! state=$(docker inspect --format '{{.State.Status}}' "$container" 2>&1); then
                printf '%-10s %s\n\n' "$container" "$state"
                continue
            fi
            printf '%-10s [%s]\n' "$container" "$state"
            if ! logs=$(docker logs --tail 200 "$container" 2>&1); then
                printf '  %s\n\n' "$logs"
                continue
            fi
            # tqdm rewrites lines with CR and can emit ANSI cursor controls.
            progress=$(printf '%s\n' "$logs" | tr '\r' '\n' |
                sed $'s/\033\\[[0-9;?]*[ -/]*[@-~]//g' |
                awk '/Progress:.*[0-9]+%\|/ { latest=$0 } END { print latest }')
            printf '  %s\n\n' "${progress:-No recent tqdm progress in the last 200 log lines.}"
        done
        [[ "$once" == true ]] && break
        sleep "$interval"
    done
)

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    watch_scrapers "$@"
fi
