# Rang buoc thoi gian. Chu ky mac dinh 10 ns (100 MHz).
# Fmax thuc te duoc TimeQuest bao cao rieng, khong bi chan boi rang buoc nay.
create_clock -name clk -period 10.000 [get_ports clk]
derive_clock_uncertainty

# Cac cong giao tiep host: khong rang buoc (giao tiep noi bo, khong phai I/O toc do cao).
set_false_path -from [get_ports rst_n] -to [all_registers]
set_false_path -from [get_ports {x_we x_addr[*] x_din[*] start}] -to [all_registers]
set_false_path -from [all_registers] -to [get_ports {done class_out[*]}]
