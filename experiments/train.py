import hydra
import torch
from hydra.utils import instantiate

from utils.utils import init_run, init_weights, get_class_weights


@hydra.main(config_path="configs", version_base=None)
def main(config):
    device = init_run(config)

    model = instantiate(config=config.model).to(device)

    if config.general.init_weights:
        model.apply(init_weights)

    if config.general.compute_class_weights:
        class_weights = get_class_weights(config.dataset.path, config.dataset.train_data, "label")
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        criterion = instantiate(config=config.loss, weight=class_weights)
    else:
        criterion = instantiate(config=config.loss)

    optimizer = instantiate(config=config.optimizer, params=model.parameters())

    train_dataloader = instantiate(config=config.dataloader, dataset_name=config.dataset.train_data).create_dataloader()
    val_dataloader = instantiate(config=config.dataloader, dataset_name=config.dataset.val_data).create_dataloader()

    trainer = instantiate(
        config=config.trainer,
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        log_every_n_steps=config.general.log_every_n_steps
    )

    for epoch in range(config.general.max_epochs):
        trainer.train(current_epoch_nr=epoch)
        trainer.evaluate(current_epoch_nr=epoch)


if __name__ == '__main__':
    main()
