#!/bin/bash

# Arrays to hold options
VARS=()
DOMAINS=()

while [[ $# -gt 0 ]]; do
	case $1 in
		--vars)
			shift
			while [[ $# -gt 0 && $1 != --* ]]; do
				VARS+=("$1")
				shift
			done
			;;
		--domains)
			shift
			while [[ $# -gt 0 && $1 != --* ]]; do
				DOMAINS+=("$1")
				shift
			done
			;;
		--timeperiod1)
			shift
			START1="$1"
			shift
			END1="$1"
			shift
			;;
		--timeperiod2)
			shift
			START2="$1"
			shift
			END2="$1"
			shift
			;;
		*)
			echo "Unknown option: $1"
			exit 1
			;;
	esac
done

# Check required args
if [[ ${#VARS[@]} -eq 0 ]]; then
	echo "Error: --vars option is required."
	exit 1
fi

if [[ ${#DOMAINS[@]} -eq 0 ]]; then
	echo "Error: --domains option is required."
	exit 1
fi

if [[ -z "$END1" ]]; then
	echo "Error: --timeperiod1 option is required."
	exit 1
fi

if [[ -z "$END2" ]]; then
	echo "Error: --timeperiod2 option is required."
	exit 1
fi

# Debug printout
echo "Variables:            ${VARS[@]}"
echo "Domains:              ${DOMAINS[@]}"
echo "First Time Period:    ${START1} to ${END1}"
echo "Second Time Period:   ${START2} to ${END2}"


# Loop through paired times and domains
for i in "${!DATES1[@]}"; do
  ts1=${DATES1[$i]}
  ts2=${DATES2[$i]}

  for dom in "${DOMAINS[@]}"; do
    f1="$NC_DIR/met_em.${dom}.${ts1}.nc"
    f2="$NC_DIR/met_em.${dom}.${ts2}.nc"

    echo " -> Swapping between $f1 and $f2"

    if [[ ! -f $f1 || ! -f $f2 ]]; then
      echo "    Warning: missing $f1 or $f2, skipping..."
      continue
    fi

    # Copy original files to swap directory
    cp "$f1" "$SWAP_DIR/"
    cp "$f2" "$SWAP_DIR/"

    # Update filenames to point to copied files
    f1_swap="$SWAP_DIR/met_em.${dom}.${ts1}.nc"
    f2_swap="$SWAP_DIR/met_em.${dom}.${ts2}.nc"

    ncks -v $VARS "$f1_swap" tmp1.nc
    ncks -v $VARS "$f2_swap" tmp2.nc
    ncks -A -v $VARS tmp1.nc "$f2_swap"
    ncks -A -v $VARS tmp2.nc "$f1_swap"
    rm tmp1.nc tmp2.nc

    # Add attribute stating a swamp happened
    ncatted -a Swapped_Variables,global,o,c,"$VARS swapped on $(date -u)" "$f1_swap"
    ncatted -a Swapped_Variables,global,o,c,"$VARS swapped on $(date -u)" "$f2_swap"
  done
done
