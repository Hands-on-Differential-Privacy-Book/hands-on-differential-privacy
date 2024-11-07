from ch04_rr_multi import make_randomized_response_multi
import opendp.prelude as dp

dp.enable_features("honest-but-curious", "contrib", "floating-point")
rr_multi = make_randomized_response_multi(p=.4, support=["A", "B", "C", "D"])

print('privately release a response of "B":', rr_multi("B"))
print('privacy expenditure ε:',               rr_multi.map(1))  # ~0.288
