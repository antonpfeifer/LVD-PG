for i in $(seq 1 400); do
  [ -d "$i" ] || echo "Missing folder: $i"
done
