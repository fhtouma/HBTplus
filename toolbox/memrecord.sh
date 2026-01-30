#!/bin/bash

jobid=$1

run_command() {

  result=$(qstat -f $jobid | sed -n '4,7p')
  date=$(date +"%Y-%m-%d %H:%M:%S")

  echo "[$date]" >> memrecord$jobid.txt
  echo "$result" >> memrecord$jobid.txt
  echo "------------------------" >> memrecord$jobid.txt
}

while true; do
  output=$(qstat $jobid)
  array=($output)
  state=${array[-2]}
  #echo $state
  if [ "$state" == "R" ]; then
    run_command
  elif [ "$state" != "Q" ]; then
    break
  fi

  # line_count=$(qstat -f $jobid | wc -l)
  # if [ $line_count -lt 7 ]; then
  #   break
  # fi
  # run_command
  sleep 600
done
