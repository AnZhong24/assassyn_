from assassyn.frontend import *
from assassyn.backend import elaborate
from assassyn import utils
import assassyn
 
class Driver(Module):

    def __init__(self):
            super().__init__(ports={})

    @module.combinational
    def build(self, record_ty: Record):
        bundle = RegArray(record_ty, 1)

        value = bundle[0].payload

        is_odd = value[0:0]
        new_value = value + Int(32)(1)
 
        new_record = record_ty.bundle(is_odd=is_odd, payload=new_value)

        # bundle[0] = new_record
 
def check_raw(raw):
    cnt = 0
    for i in raw.split('\n'):
        if 'Adder:' in i:
            line_toks = i.split()
            c = line_toks[-1]
            a = line_toks[-3]
            b = line_toks[-5]
            assert int(a) + int(b) == int(c), f'{a} + {b} != {c}'
            cnt += 1
    assert cnt == 99, f'cnt: {cnt} != 99'


def test_record():
    sys = SysBuilder('record')
    with sys:
        record_ty = Record({
            (0, 0): ('is_odd', Bits),
            (1, 32): ('payload', Int),
        })
 

        driver = Driver()
        call = driver.build( record_ty)

    print(sys)

    config = assassyn.backend.config(
            verilog=utils.has_verilator(),
            sim_threshold=200,
            idle_threshold=200,
            random=True)

    simulator_path, verilator_path = elaborate(sys, **config)

    raw = utils.run_simulator(simulator_path)
    check_raw(raw)

    if verilator_path:
        raw = utils.run_verilator(verilator_path)
        check_raw(raw)


if __name__ == '__main__':
    test_record()



