 
from assassyn.frontend import *
from opcodes import *

EPOCH_BITS = 2
EPOCH_DTYPE = Bits(EPOCH_BITS)

class MemoryAccess(Module):
    
    def __init__(self):
        super().__init__(
            ports={
                'rd': Port(Bits(5)),
                'mem_size': Port(Bits(2)),
                'mem_unsigned': Port(Bits(1)),
                'addr_lsb': Port(Bits(2)),
                'result': Port(Bits(32)),
                'is_mem_read': Port(Bits(1)),
                'epoch': Port(EPOCH_DTYPE),
            },
            no_arbiter=True)
        self.name = 'm'

    @module.combinational
    def build(
        self, 
        writeback: Module, 
        mem_bypass_reg: Array, 
        mem_bypass_data: Array,
        mem_bypass_epoch: Array,
        rdata:RegArray
    ):
        self.timing = 'systolic'

        mem_size = self.mem_size.pop()
        mem_unsigned = self.mem_unsigned.pop()
        addr_lsb = self.addr_lsb.pop()
        result = self.result.pop()
        rd = self.rd.pop()
        is_mem_read = self.is_mem_read.pop()
        epoch = self.epoch.pop()
        data = rdata[0].bitcast(Bits(32))

        is_half = mem_size == Bits(2)(1)
        is_byte = mem_size == Bits(2)(2)

        byte_data = data[0:7]
        byte_data = (addr_lsb == Bits(2)(1)).select(data[8:15], byte_data)
        byte_data = (addr_lsb == Bits(2)(2)).select(data[16:23], byte_data)
        byte_data = (addr_lsb == Bits(2)(3)).select(data[24:31], byte_data)

        half_data = addr_lsb[1:1].select(data[16:31], data[0:15])

        byte_sign = byte_data[7:7].select(Bits(24)(0xffffff), Bits(24)(0))
        half_sign = half_data[15:15].select(Bits(16)(0xffff), Bits(16)(0))

        byte_value = mem_unsigned.select(Bits(24)(0).concat(byte_data), byte_sign.concat(byte_data))
        half_value = mem_unsigned.select(Bits(16)(0).concat(half_data), half_sign.concat(half_data))
        load_value = is_byte.select(byte_value, is_half.select(half_value, data))

        with Condition(is_mem_read):
            log("mem.rdata        | 0x{:x}", data)
            log("mem.loaded       | x{:02} = 0x{:08x}", rd, load_value)
            mem_bypass_reg[0] = rd
            mem_bypass_epoch[0] = epoch

        with Condition(~is_mem_read):
            mem_bypass_reg[0] = Bits(5)(0)

        arg = is_mem_read.select(load_value, result)

        with Condition(is_mem_read & (rd != Bits(5)(0))):
            log("mem.bypass       | x{:02} = 0x{:x}", rd, arg)
            mem_bypass_data[0] = arg

        wb_bound = writeback.bind(mdata = arg , rd = rd, epoch=epoch)
        wb_bound.async_called() 
