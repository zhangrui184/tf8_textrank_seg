
import decode

_rouge_ref_dir='/home/ddd/data/cnndailymail3/finished_files/exp_logs/decode_test_400maxenc_4beam_35mindec_100maxdec_ckpt-7011/reference'
_rouge_dec_dir='/home/ddd/data/cnndailymail3/finished_files/exp_logs/decode_test_400maxenc_4beam_35mindec_100maxdec_ckpt-7011/decoded'
_decode_dir='/home/ddd/data/cnndailymail3/finished_files/exp_logs/decode'
def run():
    results_dict = decode.rouge_eval(_rouge_ref_dir, _rouge_dec_dir)
    decode.rouge_log(results_dict, _decode_dir)
run()