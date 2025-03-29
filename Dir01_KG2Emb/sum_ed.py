hit_e = [0.467886, 0.465323, 0.469658, 0.40431, 0.469087]
hit_h = [0.492616, 0.471181, 0.465792, 0.450691, 0.456792]
hit_d = [0.527543, 0.541003, 0.535497, 0.51956, 0.537588]

mr_e = [144.209095, 142.111528, 152.055647, 143.794214, 148.437705]
mr_h = [142.701828, 146.032802, 150.375703, 129.319747, 139.386885]
mr_d = [33.011721, 31.300023, 31.581186, 33.825135, 31.669906]

print('hit_e',sum(hit_e) / len(hit_e))
print('hit_h',sum(hit_h) / len(hit_h))
print('hit_d',sum(hit_d) / len(hit_d))

print('mr_e',sum(mr_e) / len(mr_e))
print('mr_h',sum(mr_h) / len(mr_h))
print('mr_d',sum(mr_d) / len(mr_d))

print('hit',(sum(hit_d)-sum(hit_e))/sum(hit_e))
print('mr',(sum(mr_d)-sum(mr_e))/sum(mr_e))
