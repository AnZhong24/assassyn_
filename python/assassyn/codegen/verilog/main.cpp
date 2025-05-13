#include <verilated.h>
#include <verilated_vcd_c.h>
#include <iostream>
#include <memory>
#include "Vtb.h"

vluint64_t sim_time = 0;

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  // For gtkwave
  Verilated::traceEverOn(true);

  auto top = std::make_unique<Vtb>();
  VerilatedVcdC* tfp = new VerilatedVcdC;
  top->trace(tfp, 99);
  tfp->open("wave.vcd");

  while (sim_time <= 10000 && !Verilated::gotFinish()) {
    if (sim_time >= 4) {
      top->rst_n = 1;
    } else {
      top->rst_n = 0;
    }

    top->clk = 1;
    top->eval();
    tfp->dump(sim_time++);

    top->clk = 0;
    top->eval();
    tfp->dump(sim_time++);
  }

  tfp->close();
  return 0;
}