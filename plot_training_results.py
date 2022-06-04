    # PLOT TRAINING
    if DISPLAY_PLOT:
        plt.figure(figsize=(8, 6))
        sns.distplot(oof_pred[-1])
        plt.show()
        
        plt.figure(figsize=(15,5))
        plt.plot(np.arange(len(history.history['auc'])),history.history['auc'],'-o',label='Train auc',color='#ff7f0e')
        plt.plot(np.arange(len(history.history['auc'])),history.history['val_auc'],'-o',label='Val auc',color='#1f77b4')
        x = np.argmax( history.history['val_auc'] ); y = np.max( history.history['val_auc'] )
        xdist = plt.xlim()[1] - plt.xlim()[0]; ydist = plt.ylim()[1] - plt.ylim()[0]
        plt.scatter(x,y,s=200,color='#1f77b4'); plt.text(x-0.03*xdist,y-0.13*ydist,'max auc\n%.2f'%y,size=14)
        plt.ylabel('auc',size=14); plt.xlabel('Epoch',size=14)
        plt.legend(loc=2)
        plt2 = plt.gca().twinx()
        plt2.plot(np.arange(len(history.history['auc'])),history.history['loss'],'-o',label='Train Loss',color='#2ca02c')
        plt2.plot(np.arange(len(history.history['auc'])),history.history['val_loss'],'-o',label='Val Loss',color='#d62728')
        x = np.argmin( history.history['val_loss'] ); y = np.min( history.history['val_loss'] )
        ydist = plt.ylim()[1] - plt.ylim()[0]
        plt.scatter(x,y,s=200,color='#d62728'); plt.text(x-0.03*xdist,y+0.05*ydist,'min loss',size=14)
        plt.ylabel('Loss',size=14)
        plt.title('FOLD %i - Image Size %i, %s'%
                (fold+1,IMG_SIZES[fold],EFNS[EFF_NETS[fold]].__name__),size=18)
        plt.legend(loc=3)
        plt.savefig(f'fig{fold}.png')
        plt.show()
