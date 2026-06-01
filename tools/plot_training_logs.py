#!/usr/bin/env python3
import json
import os
import sys
import argparse
from collections import defaultdict

try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
except Exception as e:
    print('Missing python packages. Please install pandas/numpy/matplotlib.')
    raise


def parse_logs(paths):
    train_records = []
    val_records = []
    meta = []
    for p in paths:
        with open(p, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                # some initial meta lines don't have 'mode'
                if 'mode' in obj:
                    if obj.get('mode') == 'train':
                        train_records.append(obj)
                    elif obj.get('mode') == 'val':
                        val_records.append(obj)
                else:
                    meta.append(obj)
    return train_records, val_records, meta


def coerce_numeric(df):
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        except Exception:
            pass
    return df


def find_detection_map_col(df):
    if df is None or df.empty:
        return None
    for c in df.columns:
        if 'pts_bbox' in c and 'mAP' in c:
            return c
    for c in df.columns:
        if 'mAP' in c and 'NuscMap' not in c:
            return c
    for c in df.columns:
        if 'mAP' in c:
            return c
    return None


def find_nuscmap_chamfer_col(df):
    if df is None or df.empty:
        return None
    for c in df.columns:
        if 'NuscMap_chamfer' in c and 'mAP' in c:
            return c
    # fallback
    for c in df.columns:
        if 'NuscMap_chamfer' in c:
            return c
    return None


def summarize_and_plot(train_df, val_df, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    summary = {}

    if train_df is not None and not train_df.empty:
        train_df = coerce_numeric(train_df)
        train_df = train_df.reset_index(drop=True)
        train_df['step'] = np.arange(len(train_df))
        # main training loss
        if 'loss' in train_df.columns:
            loss_col = 'loss'
        else:
            # try some alternatives
            loss_col = None
            for c in ['total_loss', 'train_loss']:
                if c in train_df.columns:
                    loss_col = c
                    break
        if loss_col is None:
            print('No training loss column found; train_df cols:', train_df.columns.tolist())
        else:
            # plot loss
            plt.figure(figsize=(10,4))
            plt.plot(train_df['step'], train_df[loss_col], alpha=0.3, label='raw')
            try:
                sm = train_df[loss_col].rolling(window=50, min_periods=1).mean()
                plt.plot(train_df['step'], sm, label='smoothed')
            except Exception:
                pass
            plt.xlabel('train step')
            plt.ylabel('loss')
            plt.title('Training loss')
            plt.legend()
            loss_png = os.path.join(out_dir, 'train_loss.png')
            plt.tight_layout()
            plt.savefig(loss_png)
            plt.close()
            summary['train_loss_plot'] = loss_png

        # lr
        if 'lr' in train_df.columns:
            plt.figure(figsize=(10,3))
            plt.plot(train_df['step'], train_df['lr'])
            plt.xlabel('train step')
            plt.ylabel('lr')
            plt.title('Learning rate')
            lr_png = os.path.join(out_dir, 'lr.png')
            plt.tight_layout()
            plt.savefig(lr_png)
            plt.close()
            summary['lr_plot'] = lr_png

        # grad_norm
        if 'grad_norm' in train_df.columns:
            plt.figure(figsize=(10,3))
            plt.plot(train_df['step'], train_df['grad_norm'])
            plt.xlabel('train step')
            plt.ylabel('grad_norm')
            plt.title('Grad norm')
            gn_png = os.path.join(out_dir, 'grad_norm.png')
            plt.tight_layout()
            plt.savefig(gn_png)
            plt.close()
            summary['grad_norm_plot'] = gn_png

        # epoch aggregated
        if 'epoch' in train_df.columns:
            ep_agg = train_df.groupby('epoch')
            if loss_col in train_df.columns:
                epoch_loss = ep_agg[loss_col].mean()
                epoch_loss.to_csv(os.path.join(out_dir, 'epoch_train_loss.csv'))
                summary['epoch_train_loss_csv'] = os.path.join(out_dir, 'epoch_train_loss.csv')

    if val_df is not None and not val_df.empty:
        val_df = coerce_numeric(val_df)
        # find detection mAP
        map_col = find_detection_map_col(val_df)
        nm_col = find_nuscmap_chamfer_col(val_df)
        if map_col is not None:
            # aggregate per epoch
            if 'epoch' in val_df.columns:
                v = val_df[['epoch', map_col]].groupby('epoch').mean()
                v = v.reset_index()
                plt.figure(figsize=(8,4))
                plt.plot(v['epoch'], v[map_col], marker='o')
                plt.xlabel('epoch')
                plt.ylabel(map_col)
                plt.title('Validation '+map_col)
                m_png = os.path.join(out_dir, 'val_map.png')
                plt.tight_layout()
                plt.savefig(m_png)
                plt.close()
                summary['val_map_plot'] = m_png
                summary['val_map_csv'] = os.path.join(out_dir, 'val_map_per_epoch.csv')
                v.to_csv(summary['val_map_csv'], index=False)

            # best val
            try:
                best_idx = val_df[map_col].idxmax()
                best_row = val_df.loc[best_idx].to_dict()
                summary['best_map'] = {map_col: float(val_df.loc[best_idx, map_col]), 'epoch': int(best_row.get('epoch', -1)), 'iter': int(best_row.get('iter', -1))}
            except Exception:
                pass
        else:
            print('No detection mAP column found in val logs. Available val columns:', val_df.columns.tolist())

        # nusc map chamfer
        if nm_col is not None and 'epoch' in val_df.columns:
            v2 = val_df[['epoch', nm_col]].groupby('epoch').mean().reset_index()
            plt.figure(figsize=(8,4))
            plt.plot(v2['epoch'], v2[nm_col], marker='o')
            plt.xlabel('epoch')
            plt.ylabel(nm_col)
            plt.title('Validation '+nm_col)
            nm_png = os.path.join(out_dir, 'val_nuscmap_chamfer.png')
            plt.tight_layout()
            plt.savefig(nm_png)
            plt.close()
            summary['val_nuscmap_plot'] = nm_png
            summary['val_nuscmap_csv'] = os.path.join(out_dir, 'val_nuscmap_per_epoch.csv')
            v2.to_csv(summary['val_nuscmap_csv'], index=False)

    # save raw csvs
    if train_df is not None and not train_df.empty:
        train_df.to_csv(os.path.join(out_dir, 'train_records.csv'), index=False)
    if val_df is not None and not val_df.empty:
        val_df.to_csv(os.path.join(out_dir, 'val_records.csv'), index=False)

    # print summary
    print('Analysis saved to:', out_dir)
    for k,v in summary.items():
        print(k+':', v)

    # print some quick stats
    if 'best_map' in summary:
        print('Best validation map:', summary['best_map'])

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('logs', nargs='+', help='log json files (ndjson)')
    parser.add_argument('-o','--outdir', default='work_dirs/analysis', help='output dir')
    args = parser.parse_args()

    train_records, val_records, meta = parse_logs(args.logs)
    print(f'Parsed {len(train_records)} train records, {len(val_records)} val records, {len(meta)} meta lines')

    train_df = pd.DataFrame(train_records)
    val_df = pd.DataFrame(val_records)

    summary = summarize_and_plot(train_df, val_df, args.outdir)


if __name__ == '__main__':
    main()
