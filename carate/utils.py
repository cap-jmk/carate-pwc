
def cv(data_set, n=5, num_epoch=150, num_classes = 2, model = Net(dim=364, dataset=dataset).to(device)):
  result = []
  acc_store = []
  auc_store = []
  loss_store = [] 
  tmp = {}
  for i in range(n):
    test_loader, train_loader, dataset, train_dataset, test_dataset  = load_data(dataset=data_set)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    for epoch in range(1, num_epoch):
        train_loss = train(epoch=epoch, model=model, device=device, optimizer=optimizer, train_loader=train_loader, num_classes = num_classes)
        loss_store.append(train_loss.cpu().tolist())
        train_acc = test(train_loader, device=device, model=model, epoch=epoch)
        test_acc = test(test_loader, device=device, model=model, epoch=epoch, test=True)
        acc_store.append([train_acc.cpu().tolist(), test_acc.cpu().tolist()])
        y = np.zeros((len(test_dataset)))
        x = np.loadtxt("MCF-7_epoch"+str(epoch)+".csv")
        for i in range(len(test_dataset)):
          y[i] = test_dataset[i].y
        y = torch.as_tensor(y)
        y = F.one_hot(y.long(), num_classes = num_classes).long()
        store_auc = []
        for i in range(len(x[0,:])): 
          auc = metrics.roc_auc_score(y[:,i], x[:,i])
          store_auc.append(auc)
        auc_store.append(store_auc)
        
        if auc >=0.9:
          break
        tmp["Loss"] = list(loss_store)
        tmp["Acc"] = list(acc_store)
        tmp["AUC"] = auc_store
    with open("/content/drive/MyDrive/CARATE_RESULTS/"+data_set+"_"+str(i)+".csv", 'w') as f:
        json.dump(tmp, f)
        print("Saved iteration one to "+"/content/drive/MyDrive/CARATE_RESULTS/"+data_set+"_"+str(i)+".csv")
    result.append(tmp)          
  return result 
