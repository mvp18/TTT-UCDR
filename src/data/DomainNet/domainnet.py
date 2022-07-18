import os
import numpy as np
import glob


def create_trvalte_splits(args):

    tr_classes = np.load(os.path.join(_BASE_PATH, 'DomainNet', 'train_classes.npy')).tolist()
    va_classes = np.load(os.path.join(_BASE_PATH, 'DomainNet', 'val_classes.npy')).tolist()
    te_classes = np.load(os.path.join(_BASE_PATH, 'DomainNet', 'test_classes.npy')).tolist()
    semantic_vec = np.load(os.path.join(_BASE_PATH, 'DomainNet', 'w2v_domainnet.npy'), allow_pickle=True, encoding='latin1').item()

    if args.trainvalid:
        tr_classes += va_classes

    all_domains = []
    for f in os.listdir(os.path.join(args.root_path, 'DomainNet')):
        if os.path.isdir(os.path.join(args.root_path, 'DomainNet', f)):
            all_domains.append(f)

    unseen_domain = args.holdout_domain
    query_domains_to_train = [args.seen_domain]
    aux_domains = np.setdiff1d(all_domains, [unseen_domain, args.seen_domain, args.gallery_domain]).tolist()
    if args.include_auxillary_domains:
        query_domains_to_train += aux_domains

    print('\nSeen:{}; Unseen:{}; Gallery:{}; Auxillary:{}.'.format(args.seen_domain, unseen_domain, args.gallery_domain, aux_domains))

    splits_gallery = trvalte_per_domain(args, args.gallery_domain, 1, tr_classes, va_classes, te_classes)
    print('{} Seen Test:{}; Unseen Test:{}'.format(args.gallery_domain, len(splits_gallery['te_seen_cls']), len(splits_gallery['te_unseen_cls'])))

    fls_train = splits_gallery['tr']
    for dom in query_domains_to_train:
        for cl in tr_classes:
            domain_cl_path = os.path.join(args.root_path, 'DomainNet', dom, cl)
            fls_train += glob.glob(os.path.join(domain_cl_path, '*.*'))
    
    fls_query_va = []

    for dom in query_domains_to_train:
        for i, cl in enumerate(va_classes):
            np.random.seed(i)
            domain_cl_path = os.path.join(args.root_path, 'DomainNet', dom, cl)
            sample_dc = glob.glob(os.path.join(domain_cl_path, '*.*'))
            random_30pc = np.random.choice(sample_dc, int(0.3*len(sample_dc)), replace=False)
            fls_query_va += random_30pc.tolist()
    
    splits = {}
    splits['tr'] = np.array(fls_train)
    splits['query_va'] = np.array(fls_query_va)
    splits['gallery_va'] = np.array(splits_gallery['va'])

    print('\n# Classes - Tr:{}; Va:{}; Te:{}'.format(len(tr_classes), len(va_classes), len(te_classes)))

    # print('#Tr Sketches:{}, #Tr Images:{}'.format(len(splits_query['tr']), len(splits_im['tr'])))
    # print('#Te Sketches:{}, #Te Images:{}'.format(len(splits_query['te']), len(splits_im['te'])))

    return {'tr_classes':tr_classes, 'va_classes':va_classes, 'te_classes':te_classes, 'semantic_vec':semantic_vec, 'splits':splits}


def trvalte_per_domain(args, domain, gzs, tr_classes, va_classes, te_classes):

    domain_path = os.path.join(args.root_path, 'DomainNet', domain)

    all_fls = np.array(glob.glob(os.path.join(domain_path, '*/*.*')))
    all_clss = np.array([f.split('/')[-2] for f in all_fls])

    fls_tr = []
    fls_va = []
    fls_te_unseen_cls = []
    fls_te_seen_cls = []

    for c in te_classes:
        fls_te_unseen_cls += all_fls[np.where(all_clss==c)[0]].tolist()
    
    for i, c in enumerate(tr_classes):

        sample_c = all_fls[np.where(all_clss==c)[0]]

        if gzs:
            np.random.seed(i)
            tr_samples = np.random.choice(sample_c, int(0.92*len(sample_c)), replace=False)
            te_seen_cls_samples = np.setdiff1d(sample_c, tr_samples)
            fls_te_seen_cls += te_seen_cls_samples.tolist()
        else:
            tr_samples = sample_c
        
        fls_tr += tr_samples.tolist()

    # for i, c in enumerate(va_classes):
    for c in va_classes:

        # sample_c = all_fls[np.where(all_clss==c)[0]]
        
        # if gzs and args.trainvalid:
        #     np.random.seed(i)
        #     va_samples = np.random.choice(sample_c, int(0.9*len(sample_c)), replace=False)
        #     te_seen_cls_samples = np.setdiff1d(sample_c, va_samples)
        #     fls_te_seen_cls += te_seen_cls_samples.tolist()
        # else:
        #     va_samples = sample_c

        # fls_va += va_samples.tolist()
        fls_va += all_fls[np.where(all_clss==c)[0]].tolist()

    splits = {}
    splits['tr'] = fls_tr
    splits['va'] = fls_va
    splits['te_seen_cls'] = fls_te_seen_cls
    splits['te_unseen_cls'] = fls_te_unseen_cls
    splits['te'] = fls_te_seen_cls + fls_te_unseen_cls

    return splits


def seen_cls_te_samples(args, domain, tr_classes, pc_per_cls=0.1):

    domain_path = os.path.join(args.root_path, 'DomainNet', domain)

    all_fls = np.array(glob.glob(os.path.join(domain_path, '*/*.*')))
    all_clss = np.array([f.split('/')[-2] for f in all_fls])

    fls_te_seen_cls = []

    for i, c in enumerate(tr_classes):

        sample_c = all_fls[np.where(all_clss==c)[0]]

        np.random.seed(i)
        te_seen_cls_samples = np.random.choice(sample_c, int(pc_per_cls*len(sample_c)), replace=False)
        fls_te_seen_cls += te_seen_cls_samples.tolist()

    return fls_te_seen_cls