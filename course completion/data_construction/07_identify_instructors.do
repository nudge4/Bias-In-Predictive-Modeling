clear
forvalues y=2014/2019 {
	foreach v in "3_spe" "4_sue" "6_fae" {
		preserve
			zipuse "C:\\Users\\ys8mz\\Box Sync\\VCCS restricted student data\\Build\\Course\\Course_`y'_`v'.dta.zip", clear
			duplicates drop
			tempfile tmp
			save "`tmp'", replace
		restore
		append using "`tmp'", force
	}
}

sort strm college subject course_num section faculty_code
bys strm college subject course_num section: keep if _n == 1
isid strm college subject course_num section
keep strm college subject course_num section faculty_code vccsid_instructor
gen full_time = faculty_code == 1
drop faculty_code
sort strm college subject course_num section
gen course = subject + "_" + course_num
drop subject course_num


sort strm college course section vccsid
replace full_time = 0 if full_time == .
preserve
	drop vccsid_instructor
	duplicates drop
	save "C:\Users\ys8mz\Box Sync\Clickstream\data\bias\full_time.dta", replace
restore
keep vccsid_instructor strm college course section
rename strm teaching_strm
duplicates drop
drop if mi(vccsid_instructor)
sort vccsid_instructor teaching_strm course course section
save "C:\Users\ys8mz\Box Sync\Clickstream\data\bias\instructors.dta", replace
keep vccsid_instructor college
duplicates drop
save "C:\Users\ys8mz\Box Sync\Clickstream\data\bias\instructor_list.dta", replace
