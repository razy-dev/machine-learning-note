model = None
criterion = None
optimizer = None
scheduler = None

if __name__ == "__main__":
    engine = RnnEngine(model, criterion, optimizer, scheduler)
    engine.train(dataloader=dataset, ephchs=10)
    engine.eval(dataloader=dataset)
