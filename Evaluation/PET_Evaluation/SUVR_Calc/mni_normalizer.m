function output = mni_normalizer(nii_file,template_file)
	spm('defaults','pet');
	spm_jobman('initcfg');
	
	disp("MATLAB START")
		
	matlabbatch{1}.spm.tools.oldnorm.estwrite.subj.source = cellstr(nii_file); % {'K:\SPM_Code\sF001.nii,1'};
	matlabbatch{1}.spm.tools.oldnorm.estwrite.subj.wtsrc = '';
	matlabbatch{1}.spm.tools.oldnorm.estwrite.subj.resample = cellstr(nii_file);  % {'K:\SPM_Code\sF001.nii,1'};
	matlabbatch{1}.spm.tools.oldnorm.estwrite.eoptions.template = cellstr(template_file);  % {'K:\spm12\spm12\toolbox\OldNorm\PET.nii,1'};
	matlabbatch{1}.spm.tools.oldnorm.estwrite.eoptions.weight = {''};
	matlabbatch{1}.spm.tools.oldnorm.estwrite.eoptions.smosrc = 8;
	matlabbatch{1}.spm.tools.oldnorm.estwrite.eoptions.smoref = 0;
	matlabbatch{1}.spm.tools.oldnorm.estwrite.eoptions.regtype = 'mni';
	matlabbatch{1}.spm.tools.oldnorm.estwrite.eoptions.cutoff = 25;
	matlabbatch{1}.spm.tools.oldnorm.estwrite.eoptions.nits = 16;
	matlabbatch{1}.spm.tools.oldnorm.estwrite.eoptions.reg = 1;
	matlabbatch{1}.spm.tools.oldnorm.estwrite.roptions.preserve = 0;
	matlabbatch{1}.spm.tools.oldnorm.estwrite.roptions.bb = [-90 -126 -72
															 90 90 108];
	matlabbatch{1}.spm.tools.oldnorm.estwrite.roptions.vox = [2 2 2];
	matlabbatch{1}.spm.tools.oldnorm.estwrite.roptions.interp = 1;
	matlabbatch{1}.spm.tools.oldnorm.estwrite.roptions.wrap = [0 0 0];
	matlabbatch{1}.spm.tools.oldnorm.estwrite.roptions.prefix = 'w';

	
	spm_jobman('run',matlabbatch);
	
	output = "Success!";

end

