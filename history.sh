sacct -a -S2023-12-05-00:00 -E2023-12-12-00:00 -X -u $1 -o jobid,start,end,time,elapsed,state | grep -P "(State|COMPLETED|---)"