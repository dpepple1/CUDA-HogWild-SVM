# /bin/sh

t=1

if [ $# -ge 2 ]; then

    while [ $t -lt 30 ]
    do
        ns=$($1 -t $t -m 2>&1 > /dev/null);
        echo "$t,$ns" >> $2;
     
        echo "Finished test with $t threads"
        t=$(($t + 1));
    done

else
    echo "Format: ./test.sh ./[executable] [output_file]"
fi

