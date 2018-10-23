import argparse
import random

def main():
    parser = argparse.ArgumentParser(description='generate AMT CSV file')
    parser.add_argument('--AMT_Q_list', type=str,
                        help="COCO annotation file", default="/home/bdong/XAI/SMQTK_example/COCO_AMT/AMT_Q_UUID.txt")
    parser.add_argument('--assignement_num', type=int,
                        help="The HIT assignment number per HIT", default=6)
    parser.add_argument('--hit_prefix', type=str,
                        default='hit_task_id_')
    parser.add_argument('--out_csv_file', type=str,
                        default='/home/bdong/Downloads/AMT_HIT.csv')

    args = parser.parse_args()

    Normal_UUID_list = list()
    Saliency_UUID_list = list()
    web_link_list = ['35.196.86.20:5000/csift', '35.196.86.20:50001/csift']
    with open(args.AMT_Q_list, 'r') as f:
        for line in f:
            Q_str = line.rstrip('\n')
            Q_wo_sa_flag = Q_str[:-1]

            Normal_UUID_list.append(Q_wo_sa_flag + '0')
            Saliency_UUID_list.append(Q_wo_sa_flag + '1')



    with open(args.out_csv_file, 'w') as f:
        title_line = "{},{}".format('google_cloud_address', ','.join(args.hit_prefix + str(i)
                                                                       for i in range(1, args.assignement_num + 1)))
        f.write(title_line)

        for i in range(len(Normal_UUID_list) * 2 // args.assignement_num):
            cur_line = "http://{}".format(web_link_list[i % 2])

            for u_idx in range(args.assignement_num // 2):
                N_uuid = random.choice(Normal_UUID_list)
                Normal_UUID_list.remove(N_uuid)
                S_uuid = random.choice(Saliency_UUID_list)

                while N_uuid[:-1] == S_uuid[:-1]:
                    S_uuid = random.choice(Saliency_UUID_list)

                Saliency_UUID_list.remove(S_uuid)

                cur_line = "{},{},{}".format(cur_line, N_uuid, S_uuid)

            f.write('\n')
            f.write(cur_line)



if __name__ == "__main__":
    main()
    print('Done')





