# /bin/sh

t=1

if [ $# -ge 2 ]; then

    while [ $t -lt 33 ]
    do
        data=$($1 -t $t -m 2>&1 > /dev/null);
        ns=$(echo $data | cut -d "," -f2)
        if [ $t -eq 1 ]; then
            first_ns=$ns
        fi

        speedup=$(echo "$first_ns / $ns" | bc -l);
        echo "$t,$data,$speedup" >> $2;
     
        echo "Finished test with $t threads"
        t=$(($t + 1));
    done

else
    echo "Format: ./test.sh ./[executable] [output_file]"
fi

