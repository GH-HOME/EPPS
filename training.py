import torch
import utils
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import time
import numpy as np
import os
import shutil


def train(model, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn,
          summary_fn, val_dataloader=None, double_precision=False, clip_grad=False, use_lbfgs=False, loss_schedules=None,
          save_state_path = None, kwargs=None, device=None):

    optim = torch.optim.Adam(lr=lr, params=model.parameters())

    if use_lbfgs:
        optim = torch.optim.LBFGS(lr=lr, params=model.parameters(), max_iter=50000, max_eval=50000,
                                  history_size=50, line_search_fn='strong_wolfe')

    numOfStage = 5
    milestones = np.linspace(0, epochs, numOfStage + 2)[1:-1].astype(np.int)
    milestones = milestones.tolist()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=milestones, gamma=0.5)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=lr, steps_per_epoch=2000, epochs=int(epochs/2000))

    if save_state_path is not None:
        utils.loadCheckpoint(save_state_path, model)


    if os.path.exists(model_dir):
        val = input("The model directory %s exists. Overwrite? (y/n)"%model_dir)
        if val == 'y':
            shutil.rmtree(model_dir)

    os.makedirs(model_dir)

    summaries_dir = os.path.join(model_dir, 'summaries')
    utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)

    writer = SummaryWriter(summaries_dir)


    total_steps = 0
    model_input = None; gt = None
    for input, gt_input in train_dataloader:
        model_input = input
        gt = gt_input

    model_input = {key: value.to(device) for key, value in model_input.items()}
    gt = {key: value.to(device) for key, value in gt.items()}

    Finish_iter_flag = False
    with tqdm(total = epochs) as pbar:
        train_losses = []
        for epoch in range(epochs):
            if not epoch % epochs_til_checkpoint and epoch:
                torch.save(model.state_dict(),
                           os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch),
                           np.array(train_losses))



            start_time = time.time()

            if double_precision:
                model_input = {key: value.double() for key, value in model_input.items()}
                gt = {key: value.double() for key, value in gt.items()}

            if use_lbfgs:
                def closure():
                    optim.zero_grad()
                    model_output = model(model_input)
                    losses = loss_fn(model_output, gt)
                    train_loss = 0.
                    for loss_name, loss in losses.items():
                        train_loss += loss.mean()
                    train_loss.backward()
                    return train_loss
                optim.step(closure)

            model_output = model(model_input)
            losses = loss_fn(model_output, gt, total_steps)

            train_loss = 0.
            for loss_name, loss in losses.items():
                single_loss = loss.mean()
                if single_loss < 0:
                    torch.save(model.state_dict(),
                               os.path.join(checkpoints_dir, 'model_best.pth'))
                    print('===================================\n==> Save best model parameters at {}'
                          .format(os.path.join(checkpoints_dir, 'model_best.pth')))

                    summary_fn(model, model_input, gt, model_output, writer, total_steps,
                               loss_val=0.0, kwargs=kwargs)
                    print('==> Save best shape recovery result at {} at iteration {}\n===================================\n'
                          .format(kwargs['save_folder'], total_steps))
                    Finish_iter_flag = True
                    break

                if loss_schedules is not None and loss_name in loss_schedules:
                    writer.add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps), total_steps)
                    single_loss *= loss_schedules[loss_name](total_steps)

                writer.add_scalar(loss_name, single_loss, total_steps)
                train_loss += single_loss

            if Finish_iter_flag:
                break

            train_losses.append(train_loss.item())
            writer.add_scalar("total_train_loss", train_loss, total_steps)

            if not total_steps % steps_til_summary:
                torch.save(model.state_dict(),
                           os.path.join(checkpoints_dir, 'model_current.pth'))

                summary_fn(model, model_input, gt, model_output, writer, total_steps, loss_val = train_loss.detach().cpu().numpy(), kwargs=kwargs)

            if not use_lbfgs:
                optim.zero_grad()
                train_loss.backward()

                if clip_grad:
                    if isinstance(clip_grad, bool):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                optim.step()
                scheduler.step()

            pbar.update(1)

            if not total_steps % steps_til_summary:
                tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))

                if val_dataloader is not None:
                    print("Running validation set...")
                    model.eval()
                    with torch.no_grad():
                        val_losses = []
                        for (model_input, gt) in val_dataloader:
                            model_output = model(model_input)
                            val_loss = loss_fn(model_output, gt)
                            val_losses.append(val_loss)

                        writer.add_scalar("val_loss", np.mean(val_losses), total_steps)
                    model.train()

            total_steps += 1

        torch.save(model.state_dict(),
                   os.path.join(checkpoints_dir, 'model_final.pth'))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
                   np.array(train_losses))


class LinearDecaySchedule():
    def __init__(self, start_val, final_val, num_steps):
        self.start_val = start_val
        self.final_val = final_val
        self.num_steps = num_steps

    def __call__(self, iter):
        return self.start_val + (self.final_val - self.start_val) * min(iter / self.num_steps, 1.)
