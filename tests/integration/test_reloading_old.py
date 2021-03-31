# For testing of the three models
from ctgan import CTGANSynthesizer, TableganSynthesizer, TVAESynthesizer
import pandas as pd
import numpy as np
import torch
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from ctgan.sampler import Sampler
from ctgan.tvae import loss_function

'''
USER INPUT IS REQUIRED HERE
- Change the modelname to test different models.
- select between ctgan, tablegan or tvae
- Note that hyperparameters are in config.py.
'''
modelname = 'tvae'  # ctgan, tablegan, tvae

"""
Sample code
"""
# Seeding.
seednum = 0
torch.manual_seed(seednum)
np.random.seed(seednum)

# Create a toy example of 3000 rows for testing
data = pd.DataFrame({
    'continuous1': np.random.random(3000),
    'discrete1': np.repeat([1, 2, 3], [2850, 75, 75]),
    'discrete2': np.repeat(["a", "b", np.nan], [1740, 1258, 2]),
    'discrete3': np.repeat([6, 7], [300, 2700])
})

# Shuffle rows
data = data.sample(frac=1).reset_index(drop=True)

# headers of discrete columns
discrete_columns = ['discrete1', 'discrete2', 'discrete3']

if modelname == 'ctgan':
    model = CTGANSynthesizer()
elif modelname == 'tablegan':
    model = TableganSynthesizer()
elif modelname == 'tvae':
    model = TVAESynthesizer()
else:
    ValueError('In valid modelname')

# Create a folder with PID to store output results
model.logger.change_dirpath(model.logger.dirpath + "/" + modelname + "_" + model.logger.PID)

# Train the model
model.fit(data, discrete_columns, model_summary=False, trans="VGM")

# Save synthetic data and trained model
timestamp = model.logger.dt.now().strftime(model.logger.datetimeformat)

model_filepath = model.logger.dirpath + "/" + modelname + "_model_" + model.logger.PID + "_" + timestamp + ".pkl"
model.save(model_filepath)

# ------------------------------------------------------------------------------------
# Reload
# ------------------------------------------------------------------------------------
print("*" * 100)
print('RELOAD AND TRAIN')

# Reload a model
model2 = torch.load(model_filepath, map_location=torch.device('cpu'))  # cuda:1

# Make changes to the toy data. Test with different data.
data2 = pd.DataFrame({
    'continuous1': np.random.random(3000)*3,
    'discrete1': np.repeat([2, 1, 3], [2850, 75, 75]),
    'discrete2': np.repeat(["b", "a", np.nan], [740, 2258, 2]),
    'discrete3': np.repeat([7, 6], [300, 2700]),
})

# # Test with similar data.
# data2 = pd.DataFrame({
#     'continuous1': np.random.random(3000),
#     'discrete1': np.repeat([1, 2, 3], [2850, 75, 75]),
#     'discrete2': np.repeat(["a", "b", np.nan], [1740, 1258, 2]),
#     'discrete3': np.repeat([6, 7], [300, 2700])
# })

# Shuffle rows
data2 = data2.sample(frac=1).reset_index(drop=True)

# headers of discrete columns
discrete_columns2 = ['discrete1', 'discrete2', 'discrete3']

temp_test_size = 15 / (70 + 15)  # 0.176
exact_val_size = int(temp_test_size * data2.shape[0])
train_data, val_data = train_test_split(data2, test_size=exact_val_size, random_state=42)

train_data = model2.transformer.transform(train_data)
val_data_transformed = model2.transformer.transform(val_data)

data_sampler = Sampler(train_data, model2.transformer.output_info, trans=model2.trans)

optimizerAE = Adam(
    list(model2.encoder.parameters()) + list(model2.decoder.parameters()), lr=model2.lr,
    weight_decay=model2.l2scale)

steps_per_epoch = max(len(train_data) // model2.batch_size, 1)

epochs = 5
trained_epoches = 0
for i in range(epochs):
    model2.decoder.train()  ##switch to train mode
    trained_epoches += 1
    for id_ in range(steps_per_epoch):
        condvec = model2.cond_generator.sample(model2.batch_size)
        if condvec is None:
            c1, m1, col, opt = None, None, None, None
            real = data_sampler.sample(model2.batch_size, col, opt)
        else:
            c1, m1, col, opt = condvec
            c1 = torch.from_numpy(c1).to(model2.device)

            perm = np.arange(model2.batch_size)
            np.random.shuffle(perm)
            real = data_sampler.sample(model2.batch_size, col[perm], opt[perm])
            c2 = c1[perm]

        optimizerAE.zero_grad()
        real = torch.from_numpy(real.astype('float32')).to(model2.device)
        if model2.cond_gen_encoder:
            real = torch.cat([real, c2], dim=1)

        mu, std, logvar = model2.encoder(real)
        eps = torch.randn_like(std)
        emb = eps * std + mu

        # NEW. 2020-12-03.
        # compute exponetial moving average
        model2.ema_mu = model2.ema_fraction * mu.mean() + (1 - model2.ema_fraction) * model2.ema_mu
        model2.ema_std = model2.ema_fraction * std.mean() + (1 - model2.ema_fraction) * model2.ema_std

        # NEW
        # Conditional vector is added to latent space.
        if model2.cond_gen_latent:
            if c1 is not None:
                emb = torch.cat([emb, c2], dim=1)
        rec, sigmas = model2.decoder(emb)
        loss_1, loss_2 = loss_function(
            rec, real, sigmas, mu, logvar, model2.transformer.output_info, model2.loss_factor,
            model2.cond_gen_encoder)
        loss = loss_1 + loss_2
        loss.backward()
        optimizerAE.step()
        model2.decoder.sigma.data.clamp_(0.01, 1.0)

    with torch.no_grad():
        model2.encoder.eval()
        model2.decoder.eval()

        if model2.cond_generator.n_opt > 0:
            c1_val = torch.zeros(
                size=(val_data_transformed.shape[0], model2.cond_generator.n_opt)).to(
                model2.device)

        this_val_data = torch.from_numpy(val_data_transformed.astype('float32')).to(model2.device)
        if model2.cond_gen_encoder:
            this_val_data = torch.cat([this_val_data, c1_val], dim=1)

        mu_val, std_val, logvar_val = model2.encoder(this_val_data)
        eps_val = torch.randn_like(std_val)
        emb_val = eps_val * std_val + mu_val

        # Conditional vector is added to latent space.
        if model2.cond_gen_latent:
            emb_val = torch.cat([emb_val, c1_val], dim=1)
        rec_val, sigmas_val = model2.decoder(emb_val)
        loss_1_val, loss_2_val = loss_function(
            rec_val, this_val_data, sigmas_val, mu_val, logvar_val,
            model2.transformer.output_info,
            model2.loss_factor, model2.cond_gen_encoder)
        val_loss = loss_1_val + loss_2_val
        model2.val_loss.append(val_loss.detach().cpu())

        model2.encoder.train()
        model2.decoder.train()

    model2.total_loss.append(loss.detach().cpu().numpy())
    model2.logger.write_to_file("Epoch " + str(trained_epoches) +
                              ", Training Loss: " + str(loss.detach().cpu().numpy()) +
                              ", Validation loss: " + str(val_loss.detach().cpu().numpy()),
                              toprint=True)

model2_filepath = model2.logger.dirpath + "/" + modelname + "_model2_" + model.logger.PID + "_" + timestamp + ".pkl"
model2.save(model2_filepath)
