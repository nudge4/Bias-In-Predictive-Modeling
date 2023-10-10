clear
forvalues y=2014/2019 {
	foreach v in 3_spe 4_sue 6_fae {
		preserve
			zipuse "C:\Users\ys8mz\Box Sync\VCCS restricted student data\Build\Class\Class_`y'_`v'.dta.zip", clear
			tempfile tmp
			save "`tmp'", replace
		restore
		append using "`tmp'", force
	}
}

keep vccsid strm college subject course_num section credit grade
duplicates drop
drop if credit == 0
replace grade = "D" if grade == "(D)"
drop if inlist(grade, "", "I", "P", "R", "S", "U", "X", "XY", "XN")
replace grade = "W" if grade == "WC"
gen course = subject + "_" + course_num
sort strm college course section vccsid
order strm college course section vccsid
drop course_num subject
sort strm college course section vccsid grade
bys strm college course section vccsid grade: egen new_credit = sum(credit)
drop credit
duplicates drop
rename new_credit credit
collapse (first) grade (sum) credit, by(strm college course section vccsid)
isid strm college course section vccsid
save "C:\Users\ys8mz\Desktop\2014_2019_vccs_courses.dta", replace
