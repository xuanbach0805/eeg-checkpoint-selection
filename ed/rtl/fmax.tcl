# Chay boi quartus_sta -t fmax.tcl <project> <revision>
project_open [lindex $quartus(args) 0] -revision [lindex $quartus(args) 1]
create_timing_netlist -model slow
read_sdc
update_timing_netlist
report_clock_fmax_summary -file fmax.txt
delete_timing_netlist
project_close
