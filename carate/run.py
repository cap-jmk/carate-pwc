device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net(dim=364).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

for epoch in range(1, 5000):
    train_loss = train(epoch)
    train_acc = test(train_loader, epoch=epoch)
    test_acc = test(test_loader, epoch, test=True)
    print(
        "Epoch: {:03d}, Train Loss: {:.7f}, "
        "Train Acc: {:.7f}, Test Acc: {:.7f}".format(
            epoch, train_loss, train_acc, test_acc
        )
    )
    # y = np.zeros((len(test_dataset)))
    # x = np.loadtxt("Enzymes_epoch"+str(epoch)+".csv")
    # for i in range(len(test_dataset)):
    # y[i] = test_dataset[i].y
    y = torch.as_tensor(y)
    y = F.one_hot(y.long(), num_classes=6).long()
    store_auc = 0
    for i in range(len(x[0, :])):
        auc = metrics.roc_auc_score(y[:, i], x[:, i])
        print("AUC of " + str(i) + "is:", auc)
        store_auc += auc
    print("Average auc", store_auc / 6)
    if auc >= 0.9:
        break
