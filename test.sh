# /bin/sh

t=1

if [ $# -ge 1 ]; then

    while [ $t -lt 30 ]
    do
        ns=$(./main -t $t -m 2>&1 > /dev/null);
        echo "$t,$ns" >> $1;
     
        echo "Finished test with $t threads"
        t=$(($t + 1));
    done

else
    echo "Please provide an output file"
fi

