clear all

root = 'D:\python_workspace\Liuhy\3Dporject\ShapeNetCore_Voxel';
voxel_root = 'D:\python_workspace\Liuhy\3Dporject\ShapeNetCore_3DVoxel';

category_file_struct = dir(root);
category_num = length(category_file_struct);

category_num=3
for i_cagegory = 1:1:category_num-2
    category_file_name = category_file_struct(i_cagegory+2).name;
    category_file_path = [root, '\', category_file_name];
    voxel_category_path = [voxel_root,'\',category_file_name];
    
    if ~exist(voxel_category_path) 
        mkdir(voxel_category_path)
    end
    
    mat_category_struct = dir( category_file_path );
    
    mat_category_num = length(mat_category_struct)
    
    for i_mat = 1:1:mat_category_num-2
        mat_file_name = mat_category_struct(i_mat+2).name;
        mat_file_path = [category_file_path,'\',mat_file_name]
        voxel_mat_path = [voxel_category_path,'\',mat_file_name]
        
        load (mat_file_path)
        Volume=polygon2voxel(FV,[64 64 64],'auto');
        eval(['save ',voxel_mat_path,' Volume'])
        
    end
    
end

%{
[row column] = size( category_file_name );

for i_category = 1:1:column
    
    
end
%}
%load obj2mat_model
%figure, patch(FV,'FaceColor',[1 0 0]); axis square;
%Volume=polygon2voxel(FV,[64 64 64],'auto');
%save voxel_model Volume