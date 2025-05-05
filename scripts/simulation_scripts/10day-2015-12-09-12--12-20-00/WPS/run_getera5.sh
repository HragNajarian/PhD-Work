#! /bin/bash -l

mamba init
mamba activate get_era5


DATE1=20151209
DATE2=20151221
Nort=21
West=79
Sout=-21
East=136


YY1=`echo $DATE1 | cut -c1-4`
MM1=`echo $DATE1 | cut -c5-6`
DD1=`echo $DATE1 | cut -c7-8`
YY2=`echo $DATE2 | cut -c1-4`
MM2=`echo $DATE2 | cut -c5-6`
DD2=`echo $DATE2 | cut -c7-8`


YY='"year": ['
for i in $(seq -f "%04g" $YY1 $YY2); do
  YY+='"'$i'", '
done
# Remove the trailing comma and space, then add the closing bracket
YY=${YY%, }']'

MM='"month": ['
for i in $(seq -f "%02g" $MM1 $MM2); do
  MM+='"'$i'", '
done
# Remove the trailing comma and space, then add the closing bracket
MM=${MM%, }']'

DD='"day": ['
for i in $(seq -f "%02g" $DD1 $DD2); do
  DD+='"'$i'", '
done
# Remove the trailing comma and space, then add the closing bracket
DD=${DD%, }']'


# sed -e "s/YY/${YY}/g;s/MM/${MM}/g;s/DD/${DD}/g;s/DATE1/${DATE1}/g;s/DATE2/${DATE2}/g;" GetERA5-sl.py > GetERA5-${DATE1}-${DATE2}-sl.py
sed -e "s/YY/${YY}/g;s/MM/${MM}/g;s/DD/${DD}/g;s/DATE1/${DATE1}/g;s/DATE2/${DATE2}/g;" GetERA5-pl.py > GetERA5-${DATE1}-${DATE2}-pl.py

# python GetERA5-${DATE1}-${DATE2}-sl.py
python GetERA5-${DATE1}-${DATE2}-pl.py



# sed -e "s/DATE1/${DATE1}/g;s/DATE2/${DATE2}/g;s/Nort/${Nort}/g;s/West/${West}/g;s/Sout/${Sout}/g;s/East/${East}/g;" GetERA5-sl.py > GetERA5-${DATE1}-${DATE2}-sl.py


# python GetERA5-${DATE1}-${DATE2}-sl.py


# sed -e "s/DATE1/${DATE1}/g;s/DATE2/${DATE2}/g;s/Nort/${Nort}/g;s/West/${West}/g;s/Sout/${Sout}/g;s/East/${East}/g;" GetERA5-pl.py > GetERA5-${DATE1}-${DATE2}-pl.py


# python GetERA5-${DATE1}-${DATE2}-pl.py

#!/bin/bash -l


# sed -e "s/YY/${YY}/g;s/MM/${MM}/g;s/DD/${DD}/g;s/Nort/${Nort}/g;s/West/${West}/g;s/Sout/${Sout}/g;s/East/${East}/g;s/DATE1/${DATE1}/g;s/DATE2/${DATE2}/g;" GetERA5-sl.py > GetERA5-${DATE1}-${DATE2}-sl.py
# sed -e "s/YY/${YY}/g;s/MM/${MM}/g;s/DD/${DD}/g;s/Nort/${Nort}/g;s/West/${West}/g;s/Sout/${Sout}/g;s/East/${East}/g;s/DATE1/${DATE1}/g;s/DATE2/${DATE2}/g;" GetERA5-pl.py > GetERA5-${DATE1}-${DATE2}-pl.py

# sed -e "s/YY/${YY}/g;s/MM/${MM}/g;s/DD/${DD}/g;s/DATE1/${DATE1}/g;s/DATE2/${DATE2}/g;" GetERA5-sl.py > GetERA5-${DATE1}-${DATE2}-sl.py
# sed -e "s/YY/${YY}/g;s/MM/${MM}/g;s/DD/${DD}/g;s/DATE1/${DATE1}/g;s/DATE2/${DATE2}/g;" GetERA5-pl.py > GetERA5-${DATE1}-${DATE2}-pl.py

# python GetERA5-${DATE1}-${DATE2}-sl.py
# python GetERA5-${DATE1}-${DATE2}-pl.py
