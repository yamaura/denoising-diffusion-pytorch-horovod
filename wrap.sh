#!/bin/bash
if [[ $OMPI_COMM_WORLD_RANK == 0 ]]; then
   nsys profile -o /tmp/report-%p "$@"
else
   "$@"
fi
